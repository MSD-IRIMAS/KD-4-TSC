"""" Knowledge Distillation implementation in PyTorch"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation."""
    
    def __init__(self, alpha=0.3, temperature=10.0):
        """
        Args:
            alpha: Weight for student loss (1-alpha for distillation loss)
            temperature: Temperature for softening probability distributions
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss_fn = nn.CrossEntropyLoss()
        self.distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: Raw logits from student model (no softmax)
            teacher_logits: Raw logits from teacher model (no softmax)
            labels: Ground truth labels (one-hot or class indices)
            
        Returns:
            total_loss: Combined loss
            student_loss: Cross-entropy loss
            distillation_loss: KL divergence loss
        """
        
        # Student loss (cross-entropy)
        if labels.dim() > 1: # One-hot encoded
            labels = labels.argmax(dim=1)
        
        student_loss = self.student_loss_fn(student_logits, labels)
        
        
        # Distillation loss (KL divergence)
        # Soften predictions with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distillation_loss =  self.distillation_loss_fn(student_soft, teacher_soft)
        
        # Combined losses
        total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        return total_loss, student_loss, distillation_loss
    
    
    
class Distiller:
    """ Knowledge Distillation trainer."""
    
    def __init__(self, student, teacher, alpha=0.3, temperature=10.0, device='cuda'):
        """
        Args:
            student: Student model
            teacher: Teacher model (should be pre-trained)
            alpha: Weight for student loss
            temperature: Temperature for distillation
            device: Device to train on
        """
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.device = device
        
        # Freeze the teacher model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        
        # Loss function
        self.criterion = DistillationLoss(alpha=alpha, temperature=temperature)
        
    
    def train_step(self, batch_data, optimizer):
        """
        Perform one training step.
        Args:
            batch_data: Tuple of (inputs, labels)
            optimizer: Optimizer
            
        Returns:
            Dictionary of losses and metrics
        """
        
        inputs, labels = batch_data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Set student to training mode
        self.student.train()
        
        # Get teacher predictinos (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            
        # Forward pass through student
        student_logits = self.student(inputs)
        
        # Compute loss
        total_loss, student_loss, distillation_loss = self.criterion(
            student_logits, teacher_logits, labels
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Compute accuracy
        if labels.dim() > 1: # One-hot encoded
            labels = labels.argmax(dim=1)
        
        predictinos = student_logits.argmax(dim=1)
        accuracy = (predictinos == labels).float().mean()
        
        return {
            'total_loss': total_loss.item(),
            'student_loss': student_loss.item(),
            'distillation_loss': distillation_loss.item(),
            'accuracy': accuracy.item()
        }
        
        
    def evaluate(self, data_loader):
        """
        Evaluate student model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Dictionary of average metrics
        """
        self.student.eval()
        
        total_loss = 0.0
        total_student_loss = 0.0
        total_distillation_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                teacher_logits = self.teacher(inputs)
                student_logits = self.student(inputs)
                
                # Compute loss
                total_loss_batch, student_loss_batch, distillation_loss_barch = \
                    self.criterion(student_logits, teacher_logits, labels)
                    
                # Compute accuracy
                if labels.dim() > 1:
                    labels = labels.argmax(dim=1)
                predictinos = student_logits.argmax(dim=1)
                accuracy = (predictinos == labels).float().mean()
                
                # Accumulate metrics
                total_loss += total_loss_batch.item()
                total_student_loss += student_loss_batch.item()
                total_distillation_loss += distillation_loss_barch.item()
                total_accuracy += accuracy
                num_batches += 1
                
        return {
            'total_loss': total_loss / num_batches,
            'student_loss': total_student_loss / num_batches,
            'distillation_loss': total_distillation_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
                
        