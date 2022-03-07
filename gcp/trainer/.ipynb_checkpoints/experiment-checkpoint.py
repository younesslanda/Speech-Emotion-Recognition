#Author : Youness Landa
import copy, time, logging

from config import Config as cfg

class Experiment:
    def __init__(self, dataloader_train, dataloader_valid, optimizer, criterion, writer):
        self.dataloaders = dict()
        self.dataloaders['train'] = dataloader_train
        self.dataloaders['valid'] = dataloader_valid
        
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.writer = writer
        
    def run(self, model):
        
        best_model_weights = copy.deepcopy(model.state_dict())
        best_accuracy = 0.0
        
        start_train = time.time()
        logging.info(' - Start of training of the model - \n')
        
        for epoch in range(cfg.NUM_EPOCHS):
            logging.info('\nEpoch {}/{}'.format(epoch + 1, cfg.NUM_EPOCHS))
            logging.info('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                    
                running_loss = 0.0
                running_emotion_corrects = 0
                running_gender_corrects = 
                
                for i_batch, sample_batched in enumerate(self.dataloaders[phase]):
                    features, lengths, emotion_idxs, gender_idxs = sample_batched

                    self.optimizer.zero_grad()
                    
                    # forward pass
                    # if in train phase, we enable grad
                    with torch.set_grad_enabled(phase == 'train'):
                        emotion_output, gender_output = model(features, lengths)
                        
                        _, emotion_predictions = torch.max(emotion_output, 1)
                        _, gender_predictions  = torch.max(gender_output, 1)

                        emotion_loss = self.criterion(emotion_predictions, emotion_idxs)
                        gender_loss  = self.criterion(gender_predictions, gender_idxs)

                        total_loss = cfg.ALPHA * emotion_loss + cfg.BETA * gender_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            total_loss.backward()
                            self.optimizer.step() 
                            
                    # statistics
                    running_loss += total_loss.item() * features.size(0)
                    running_emotion_corrects += torch.sum(emotion_predictions == emotion_idxs)
                    running_gender_corrects  += torch.sum(gender_predictions  == gender_idxs)
                
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_accuracy_emotion = running_emotion_corrects / len(self.dataloaders[phase].dataset)
                epoch_accuracy_gender  = running_gender_corrects  / len(self.dataloaders[phase].dataset)
                
                if phase == 'train':
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/train/E', epoch_accuracy_emotion, epoch)
                    writer.add_scalar('Accuracy/train/G', epoch_accuracy_gender, epoch)

                elif phase == 'valid':
                    writer.add_scalar('Loss/valid', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/valid/E', epoch_accuracy_emotion, epoch)
                    writer.add_scalar('Accuracy/valid/G', epoch_accuracy_gender, epoch)
                    
                print('phase : {} --- epoch : {} -- Loss: {:.4f} - Acc/E: {:.4f} - Acc/G: {:.4f}'.format(
                    phase, epoch, epoch_loss, epoch_accuracy_emotion, epoch_accuracy_gender))
                logging.info('phase : {} --- epoch : {} -- Loss: {:.4f} - Acc/E: {:.4f} - Acc/G: {:.4f}'.format(
                    phase, epoch, epoch_loss, epoch_accuracy_emotion, epoch_accuracy_gender))
                
                # deep copy the best model
                if phase == 'valid' and epoch_accuracy_emotion > best_accuracy:
                    best_accuracy = epoch_accuracy_emotion
                    best_model_weights = copy.deepcopy(model.state_dict())
                    
        end_train = time.time() - start_train
        print('\n\nTraining complete in {:.0f}m {:.0f}s'.format(end_train // 60, end_train % 60))
        print('Best validation emotion accuracy: {:4f}'.format(best_accuracy))
        
        logging.info('/nTraining complete in {:.0f}m {:.0f}s'.format(end_train // 60, end_train % 60))
        logging.info('Best validation emotion accuracy: {:4f}'.format(best_accuracy))
        
        model.load_state_dict(best_model_weights)
        return model