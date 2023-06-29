import h5py
import importlib 
import tensorflow as tf

def evaluate_loss(model, dataset, loss_func):
    avg_loss = tf.keras.metrics.Mean(dtype=tf.float32)
    for image, label in dataset:
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


def get_tf_datasets(data_file, batch_size, shuffle_buffer): #, train_size):
    datasets = dict()
    print('Loading data...')
    with h5py.File(data_file, 'r') as f:
        for phase in ['train', 'valid']:
            #data = f[phase][:train_size] if phase == 'train' else f[phase][:]
            data = f[phase][:]
            input_data = tf.data.Dataset.from_tensor_slices(data[...,:-1].astype('float32'))
            label_data = tf.data.Dataset.from_tensor_slices(data[...,-1:].astype('float32'))
    
            datasets[phase] = tf.data.Dataset.zip((input_data, label_data)) \
                    .shuffle(shuffle_buffer) \
                    .batch(batch_size) \
                    .prefetch(tf.data.AUTOTUNE)
    return datasets


def get_TFRecord_dataset(data_file, shuffle_buffer, batch_size):
    print('Loading data...')
    # load data
    def _parse_example(example_string):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(example_string, feature_description)
                
        image = tf.io.decode_raw(features['image'], tf.float32)
        label = tf.io.decode_raw(features['label'], tf.float32)
        image = tf.reshape(image, [640, 80, 3]) 
        label = tf.reshape(label, [640 ,80, 1])
        return image, label

    raw_dataset = tf.data.TFRecordDataset(data_file)
    dataset = raw_dataset.map(_parse_example)
    
    # split to train/valid
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    count = sum(1 for _ in dataset)
    train_size = int(count*0.7)
    ds_for_model ={
        'train' : dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE),
        'valid' : dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    }
    return ds_for_model


def create_model_by_exp_settings(model_name, load_from=''):
    tf.keras.backend.clear_session()
    print('Create model...')
    model_class = importlib.import_module(f'models.{model_name}').Model()
    if load_from:
        model_class.load_weights(load_from).expect_partial()
    return model_class

def initial_tf_variables():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
