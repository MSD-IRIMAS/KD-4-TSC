"""Trainer class for model training and evaluation."""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score

from models.inception import Inception
from models.fcn import FCN
from models.convtran import ConvTran
from distiller import Distiller


class Trainer:
    """Trainer for Inception and FCN models."""

    def __init__(self, model_type, input_shape, nb_classes, output_dir,
                 epochs=1500, batch_size=64, lr=0.001, patience=50,
                 min_lr=0.0001, lr_factor=0.5, device='cuda',
                 teacher_path=None, alpha=0.3, temperature=10.0,
                 architecture='inception',
                 # Inception parameters
                 teacher_depth=6, student_depth=4, nb_filters=32,
                 bottleneck_size=32, kernel_size=40,
                 # FCN parameters
                 teacher_filters=[128, 256, 128], student_filters=[64, 128, 64],
                 teacher_kernel_sizes=None, student_kernel_sizes=None,
                 # ConvTran parameters
                 teacher_num_heads=8, student_num_heads=6,
                  ):
        """
        Args:
            model_type: 'teacher', 'student_kd', or 'student_alone'
            input_shape: Shape of input (seq_length, n_features)
            nb_classes: Number of classes
            output_dir: Directory to save outputs
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            patience: Patience for learning rate scheduler
            min_lr: Minimum learning rate
            lr_factor: Factor to reduce learning rate
            device: Device to train on
            teacher_path: Path to teacher model (for KD)
            aplha: Weight for student loss (for KD)
            temperature: Temperature for distillation (for KD)
            architecture: 'inception' or 'fcn'
            
            # Inception-specific
            teacher_depth: Depth of teacher Inception model
            student_depth: Depth of student Inception model
            nb_filters: Number of filters for Inception
            bottleneck_size: Bottleneck size for Inception
            kernel_size: Kernel size for Inception
            
            # FCN-specific
            teacher_filters: Filter sizes for teacher FCN (length = depth)
                           Example: [128, 256, 128] = 3 layers
            student_filters: Filter sizes for student FCN (length = depth)
                           Example: [64, 128] = 2 layers, compressed depth+width
            teacher_kernel_sizes: Optional kernel sizes for teacher FCN
            student_kernel_sizes: Optional kernel sizes for student FCN

        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.architecture = architecture.lower()

        os.makedirs(output_dir, exist_ok=True)

        # Build model based on architecture and type
        if self.architecture == 'inception':
            self.model = self._build_inception_model(
                model_type, input_shape, nb_classes, teacher_depth, student_depth,
                nb_filters, bottleneck_size, kernel_size, teacher_depth, alpha, temperature, device
            )
        elif self.architecture == 'fcn':
            self.model = self._build_fcn_model(
                model_type, input_shape, nb_classes, teacher_filters, student_filters,
                teacher_kernel_sizes, student_kernel_sizes,
                teacher_path, alpha, temperature, device
            )
        elif self.architecture == 'convtran':
            self.model = self._build_convtran_model(model_type, input_shape, teacher_num_heads, student_num_heads, nb_classes, 
                                                    teacher_path, alpha, temperature, device)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Choose 'inception', 'fcn' or 'convtran'.")

        # Training components (already set up in build methods for KD)
        if model_type != 'student_kd':
            self.model = self.model.to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.criterion = nn.CrossEntropyLoss()
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                               factor=lr_factor, patience=patience,
                                               min_lr=min_lr)
            self.is_distiller = False


        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        if self.is_distiller:
            self.history['student_loss'] = []
            self.history['distillation_loss'] = []
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    
    def _build_inception_model(self, model_type, input_shape, nb_classes, 
                               teacher_depth, student_depth, nb_filters, 
                               bottleneck_size, kernel_size, 
                               teacher_path, alpha, temperature, device):
        """Build Inception model."""
        if model_type == 'teacher':
            return Inception(
                input_shape=input_shape,
                nb_classes=nb_classes,
                depth=teacher_depth,
                nb_filters=nb_filters,
            )
        elif model_type == 'student_alone':
            return Inception(
                input_shape=input_shape,
                nb_classes=nb_classes,
                depth=student_depth,
                nb_filters=nb_filters,
            )
        elif model_type == 'student_kd':
            # Load teacher
            teacher = Inception(
                input_shape=input_shape,
                nb_classes=nb_classes,
                depth=teacher_depth,
                nb_filters=nb_filters,
            )

            teacher.load_state_dict(torch.load(teacher_path, map_location=device))

            # Create student
            student = Inception(
                input_shape=input_shape,
                nb_classes=nb_classes,
                depth=student_depth,
                nb_filters=nb_filters,
            )
            
            
            # Create distiller
            self.distiller = Distiller(student, teacher, alpha, temperature, device)
            self.model = self.distiller.student
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                              factor=0.5, patience=50, min_lr=0.0001)
            self.is_distiller = True

            return self.model
    
    def _build_fcn_model(self, model_type, input_shape, nb_classes,
                        teacher_filters, student_filters,
                        teacher_kernel_sizes, student_kernel_sizes,
                        teacher_path, alpha, temperature, device):
        """Build FCN model with flexible depth and width."""
        if model_type == 'teacher':
            kwargs = {'filters': teacher_filters}
            if teacher_kernel_sizes is not None:
                kwargs['kernel_sizes'] = teacher_kernel_sizes
            return FCN(
                input_shape=input_shape,
                nb_classes=nb_classes,
                **kwargs
)
        elif model_type == 'student_alone':
            kwargs = {'filters': student_filters}
            if student_kernel_sizes is not None:
                kwargs['kernel_sizes'] = student_kernel_sizes
            return FCN(
                input_shape=input_shape,
                nb_classes=nb_classes,
                **kwargs
            )
        elif model_type == 'student_kd':
            # Load teacher
            teacher_kwargs = {'filters': teacher_filters}
            if teacher_kernel_sizes is not None:
                teacher_kwargs['kernel_sizes'] = teacher_kernel_sizes
            teacher = FCN(
                input_shape=input_shape,
                nb_classes=nb_classes,
                **teacher_kwargs
            )
            teacher.load_state_dict(torch.load(teacher_path, map_location=device))
            
            # Create student
            student_kwargs = {'filters': student_filters}
            if student_kernel_sizes is not None:
                student_kwargs['kernel_sizes'] = student_kernel_sizes
            student = FCN(
                input_shape=input_shape,
                nb_classes=nb_classes,
                **student_kwargs
            )
            
            # Create distiller
            self.distiller = Distiller(student, teacher, alpha, temperature, device)
            self.model = self.distiller.student
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                              factor=0.5, patience=50, min_lr=0.0001)
            self.is_distiller = True
            
            return self.model

    def _build_convtran_model(self, model_type, input_shape, teacher_num_heads, student_num_heads, nb_classes, teacher_path, alpha, temperature, device):
        """Build ConvTran model."""
        if model_type == 'teacher':
            return ConvTran(input_shape, teacher_num_heads, nb_classes)
        
        elif model_type == 'student_alone':
            kwargs = {'num_heads': student_num_heads}
            return ConvTran(input_shape, nb_classes, **kwargs)
        
        elif model_type == 'student_kd':
            # Load teacher
            teacher_kwarsg = {'num_heads': teacher_num_heads}
            teacher = ConvTran(input_shape, nb_classes, **teacher_kwarsg)
            teacher.load_state_dict(torch.load(teacher_path, map_location=device))

            # Create student
            student_kwargs = {'num_heads': student_num_heads}
            student = ConvTran(input_shape, nb_classes, **student_kwargs)
            
            # Create distiller
            self.distiller = Distiller(student, teacher, alpha, temperature, device)
            self.model = self.distiller.student
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                              factor=0.5, patience=50, min_lr=0.0001)
            self.is_distiller = True

            return self.model



    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if self.is_distiller:
            total_student_loss = 0.0
            total_distillation_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # print(f"Batch size: {inputs.size(0)}, Input shape: {inputs.shape}, Labels shape: {labels.shape}")
            # Get labels as indices for accuracy computation
            if labels.dim() > 1:  # One-hot encoded
                labels_idx = labels.argmax(dim=1)
            else:
                labels_idx = labels
            
            if self.is_distiller:
                # Knowledge distillation training
                metrics = self.distiller.train_step((inputs, labels), self.optimizer)
                
                # Accumulate losses
                batch_size = inputs.size(0)
                total_loss += metrics['total_loss'] * batch_size
                total_student_loss += metrics['student_loss'] * batch_size
                total_distillation_loss += metrics['distillation_loss'] * batch_size

                # Compute accuracy for this batch
                with torch.no_grad():
                    student_logits = self.distiller.student(inputs)
                    predictions = student_logits.argmax(dim=1)
                    total_correct += (predictions == labels_idx).sum().item()
                
                total_samples += inputs.size(0)
            
            else:
                # Standard training
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels_idx)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions == labels_idx).sum().item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        result = {'loss': avg_loss, 'accuracy': avg_acc}

        if self.is_distiller:
            result['student_loss'] = total_student_loss / total_samples
            result['distillation_loss'] = total_distillation_loss / total_samples

        return result
    

    def evaluate(self, data_loader):
        """Evaluate model on dataset."""
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                
                # Handle labels
                if labels.dim() > 1:  # One-hot encoded
                    labels_idx = labels.argmax(dim=1)
                else:
                    labels_idx = labels
                
                loss = nn.CrossEntropyLoss()(outputs, labels_idx)
                
                total_loss += loss.item() * inputs.size(0)
                predictions = outputs.argmax(dim=1)
                total_correct += (predictions == labels_idx).sum().item()
                total_samples += inputs.size(0)
    
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': avg_acc}



    def fit(self, train_loader, val_loader=None):
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        print(f'\nTraining {self.model_type} model...')
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if self.is_distiller:
                self.history['student_loss'].append(train_metrics['student_loss'])
                self.history['distillation_loss'].append(train_metrics['distillation_loss'])

            # Learning rate scheduler
            self.scheduler.step(train_metrics['loss'])

            # Save best model (based on training loss)
            if train_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = train_metrics['loss']
                self.best_epoch = epoch
                self.save_model(os.path.join(self.output_dir, 'best_model.pth'))

            # Print progress
            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs} - "
                      f"Loss: {train_metrics['loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
        
        duration = time.time() - start_time
        print(f"\nTraining completed in {duration:.2f}s")
        print(f"Best model at epoch {self.best_epoch} with loss {self.best_val_loss:.4f}")

        return duration
    

    
    def save_model(self, path):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_history(self):
        """Save training history to CSV."""
        hist_df = pd.DataFrame(self.history)
        hist_df.to_csv(os.path.join(self.output_dir, 'history.csv'), index=False)

                
