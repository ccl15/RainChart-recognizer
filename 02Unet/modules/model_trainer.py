import tensorflow as tf
from collections import defaultdict
from modules.training_helper import evaluate_loss
import os, glob

def train_model(
    model,
    datasets,
    summary_writer,
    saving_path,
    batch_size,
    evaluate_freq,
    max_epoch,
    loss_name,
    L_rate,
    overfit_stop=None
):
    if loss_name == 'MSE':
        loss_function = tf.keras.losses.MeanSquaredError()
    elif loss_name == 'MAE':
        loss_function = tf.keras.losses.MeanAbsoluteError()
    elif loss_name == 'BC':
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        
    optimizer = tf.keras.optimizers.Adam( float(L_rate) ) # prepare optimizer
    avg_losses = defaultdict(lambda: tf.keras.metrics.Mean(dtype=tf.float32))

    @tf.function
    def train_step(image, label, training=True):
        with tf.GradientTape() as tape:
            predict = model(image, training=training)
            pred_loss = loss_function(predict, label)
        gradients = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        avg_losses[f'{loss_name}'].update_state(pred_loss)
        return
    

    best_losses = [9e10, 9e10, 9e10]
    best_epochs = [0, 0, 0]
    for epoch_index in range(1, max_epoch+1):
        # ---- train
        print('Epoch #%d' % (epoch_index))
        for image, label in datasets['train'].batch(batch_size):
            train_step(image, label, training=True)

        for loss_name, avg_loss in avg_losses.items():
            with summary_writer.as_default():
                tf.summary.scalar(loss_name, avg_loss.result(), step=epoch_index)
            avg_loss.reset_states()

        # ---- evaluate
        if (epoch_index) % evaluate_freq == 0:
            print(f'Evaluate epochs {epoch_index}')
            
            for phase in ['train', 'valid']:
                loss  = evaluate_loss(model, datasets[phase].batch(batch_size), loss_function)
                with summary_writer.as_default():
                    tf.summary.scalar(f'[{phase}]: {loss_name}', loss, step=epoch_index) 
             
            valid_loss = loss
            if valid_loss < max(best_losses):
                replaced_idx = best_losses.index(max(best_losses))
                old_epoch = best_epochs[replaced_idx]
                best_losses[replaced_idx] = valid_loss
                best_epochs[replaced_idx] = epoch_index
                print(f'Get best three loss. Save epoch {epoch_index}.')
                model.save_weights(f'{saving_path}/M{epoch_index:03d}', save_format='tf')
                
                if old_epoch > 0:
                    fs = glob.glob(f'{saving_path}/M{old_epoch:03d}*')
                    print(f'Remove {old_epoch}')
                    for f in fs:
                        os.remove(f)
            elif overfit_stop and (epoch_index - max(best_epochs)) >= overfit_stop:
                print('overfiting early stop!')
                break
            
