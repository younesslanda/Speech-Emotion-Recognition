#Author : Youness Landa

from config import Config as cfg

class Experiment:
    def __init__(self, dataloader_train, optimizer, criterion):
        self.dataloader_train = dataloader_train
        self.optimizer = optimizer
        self.criterion = criterion
        
    def run(self, model):
        for epoch in range(cfg.NUM_EPOCHS):
            for i_batch, sample_batched in enumerate(self.dataloader_train):
                features, lengths, emotion_idxs, gender_idxs = sample_batched
                
                self.optimizer.zero_grad()
                
                emotion_predictions, gender_predictions = model(features, lengths)
                
                emotion_loss = self.criterion(emotion_predictions, emotion_idxs)
                gender_loss = self.criterion(gender_predictions, gender_idxs)
                
                total_loss = cfg.ALPHA * emotion_loss + cfg.BETA * gender_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                print('Epoch {}/{}'.format(epoch, epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                  if phase == 'train':
                    model.train()  # Set model to training mode
                  else:
                    model.eval()   # Set model to evaluate mode
                
        