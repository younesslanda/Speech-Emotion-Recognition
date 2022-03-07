#Author : Youness Landa
import copy, time, logging

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from config import Config as cfg

class Experiment:
    def __init__(self, dataloader_train, dataloader_valid, dataloader_test, optimizer, criterion, writer):
        self.dataloaders = dict()
        self.dataloaders['train'] = dataloader_train
        self.dataloaders['valid'] = dataloader_valid
        self.dataloaders['test']  = dataloader_test
        
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
                    self.writer.add_scalar('Loss/train', epoch_loss, epoch)
                    self.writer.add_scalar('Accuracy/train/E', epoch_accuracy_emotion, epoch)
                    self.writer.add_scalar('Accuracy/train/G', epoch_accuracy_gender, epoch)

                elif phase == 'valid':
                    self.writer.add_scalar('Loss/valid', epoch_loss, epoch)
                    self.writer.add_scalar('Accuracy/valid/E', epoch_accuracy_emotion, epoch)
                    self.writer.add_scalar('Accuracy/valid/G', epoch_accuracy_gender, epoch)
                    
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
    
    def test(self, model):
        predictions_emotion = []
        predictions_gender  = []
        
        logging.info('\n- Beginning the test -')
        for i_, sample in enumerate(self.dataloaders['test']):
            feature, length, emotion_idx, gender_idx = sample
            with torch.no_grad():
                emotion_output, gender_output = model(feature, length)
                
                _, emotion_prediction = torch.max(emotion_output, 1)
                _, gender_prediction  = torch.max(gender_output, 1)
                
                predictions_emotion.append((emotion_idx.item(), emotion_prediction.item()))
                predictions_gender.append((gender_idx.item(), gender_prediction.item()))
                logging.info('true label [E | G] : [{} | {}] --- predicted label [E | G] : [{} | {}]'.format(
                   emotion_idx.item(), gender_idx.item(), emotion_prediction.item(), gender_prediction.item()
                ))
        
        #from list of tuples to tuple of lists
        true_emotion, pred_emotion = zip(*predictions_emotion)
        true_gender , pred_gender  = zip(*predictions_gender)
        
        logging.info('\nCalculating the confusion matrices')
        emotion_cm = confusion_matrix(true_emotion, pred_emotion, labels=cfg.EMOTION_LABELS)
        gender_cm  = confusion_matrix(true_gender , pred_gender, labels=cfg.GENDER_LABELS)
        
        logging.info('Generating the classification report')
        emotion_classification_report = classification_report(true_emotion, pred_emotion, target_names=cfg.EMOTION_NAMES, 
                                                              labels=cfg.EMOTION_LABELS)
        gender_classification_report  = classification_report(true_gender , pred_gender, target_names=cfg.GENDER_NAMES
                                                             labels=cfg.GENDER_LABELS)
        
        logging.info('Writing the confusion matrix figure of the test set to TensorBoard ')
        logging.info('confusion matrix - emotion')
        plt.figure(figsize = (10,7))
        figure = sns.heatmap(emotion_cm, annot=True, cmap='YlGn').get_figure()
        plt.close(figure)
        self.writer.add_figure("Test/E/cm", figure)
        
        logging.info('confusion matrix - gender')
        plt.figure(figsize = (10,7))
        figure = sns.heatmap(gender_cm, annot=True, cmap='YlGn').get_figure()
        plt.close(figure)
        self.writer.add_figure("Test/G/cm", figure)
        
        logging.info('Writing the classification report to TensorBoard ')
        logging.info('classification report - emotion')
        self.writer.add_text('Test/E/report', emotion_classification_report)
        logging.info('classification report - gender')
        self.writer.add_text('Test/G/report', gender_classification_report)