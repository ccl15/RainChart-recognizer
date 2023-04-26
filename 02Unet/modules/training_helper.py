import h5py
import importlib 
import tensorflow as tf

def evaluate_loss(model, dataset, loss_func):
    avg_loss = tf.keras.metrics.Mean(dtype=tf.float32)
    for batch_index, (image, label) in dataset.enumerate():
        pred = model(image, training=False)
        loss = loss_func(label, pred)
        avg_loss.update_state(loss)
    return avg_loss.result()

# tf ------------------------
def set_up_tensorflow(GPU_limit):
    # shut up tensorflow!
    tf.get_logger().setLevel('ERROR')
    # restrict the memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_limit)]
        )
    
    
def get_summary_writer(log_path):
    return tf.summary.create_file_writer(log_path)


def get_tf_datasets(data_file, batch_size, shuffle_buffer):
    datasets = dict()
    print('Loading data...')
    with h5py.File(data_file, 'r') as f:
        for phase in ['train', 'valid']:
            data = f[phase][:]
            input_data = tf.data.Dataset.from_tensor_slices(data[...,:-1].astype('float32'))
            label_data = tf.data.Dataset.from_tensor_slices(data[...,-1:].astype('float32'))
    
            datasets[phase] = tf.data.Dataset.zip((input_data, label_data)) \
                    .shuffle(shuffle_buffer) \
                    .batch(batch_size)
    return datasets


def create_model_by_exp_settings(model_name, load_from=''):
    tf.keras.backend.clear_session()
    print('Create model...')
    model_class = importlib.import_module(f'models.{model_name}').Model()
    if load_from:
        model_class.load_weights(load_from).expect_partial()
    return model_class
