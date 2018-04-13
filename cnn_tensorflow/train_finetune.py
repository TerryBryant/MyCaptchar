import tensorflow as tf

# captcha infos
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

char_set = number + alphabet
CHAR_SET_LEN = len(char_set)

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags
flags.DEFINE_integer('num_epochs', 50, 'Number of traning epochs')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_float('dropout_rate', 0.75, 'Dropout rate')
flags.DEFINE_string('train_dataset', 'tfrecords_file/captcha_train.tfrecords', 'Filename of train dataset')  # 注意训练集的目录
flags.DEFINE_string('valid_dataset', 'tfrecords_file/captcha_valid.tfrecords', 'Filename of valid dataset')
flags.DEFINE_string('warm_model_dir', 'trained_model/lenet_captcha', 'Finetune model path')  # 待finetune的模型路径
flags.DEFINE_string('model_dir', 'trained_model/lenet_captcha2', 'Filename of model ')  # 注意文件保存路径

flags.DEFINE_integer('CHAR_SET_LEN', CHAR_SET_LEN, 'Range of the words in captcha')
flags.DEFINE_integer('MAX_CAPTCHA', 4, 'Lengh of the captcha')
flags.DEFINE_integer('IMAGE_HEIGHT', 34, 'Height of the captcha image')
flags.DEFINE_integer('IMAGE_WIDTH', 66, 'Width of the captcha image')
flags.DEFINE_integer('IMAGE_CHANNELS', 1, 'Channels of the captcha image')
FLAGS = flags.FLAGS


# 定义模型函数
# 该函数需要返回一个定义好的tf.estimator.EstimatorSpec对象，对于不同的mode，提供的参数不一样
# 训练模式，即 mode == tf.estimator.ModeKeys.TRAIN，必须提供的是 loss 和 train_op。
# 验证模式，即 mode == tf.estimator.ModeKeys.EVAL，必须提供的是 loss。
# 预测模式，即 mode == tf.estimator.ModeKeys.PREDICT，必须提供的是 predicitions。
def lenet_model_fn(features, labels, mode):
    # 输入层
    x = tf.reshape(features, shape=[-1, FLAGS.IMAGE_HEIGHT, FLAGS.IMAGE_WIDTH, FLAGS.IMAGE_CHANNELS], name="x_input")

    # 卷积层1
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3],
                         padding='same', activation=tf.nn.relu, name='conv1')
    # 池化层1
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2,
                                padding='same', name='pool1')
    # drop out1
    x = tf.layers.dropout(inputs=x, rate=FLAGS.dropout_rate, name='dropout1')

    # 卷积层2
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3],
                         padding='same', activation=tf.nn.relu, name='conv2')
    # 池化层2
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2,
                                padding='same', name='pool2')
    # drop out2
    x = tf.layers.dropout(inputs=x, rate=FLAGS.dropout_rate, name='dropout2')

    # 卷积层3
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[3, 3],
                         padding='same', activation=tf.nn.relu, name='conv3')
    # 池化层3
    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2,
                                padding='same', name='pool3')
    # drop out3
    x = tf.layers.dropout(inputs=x, rate=FLAGS.dropout_rate, name='dropout3')

    # 全连接层1
    x = tf.reshape(x, [-1, 5 * 9 * 128])
    x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu, name='dense')

    # drop out3
    x = tf.layers.dropout(inputs=x, rate=FLAGS.dropout_rate, name='dropout4')

    logits = tf.layers.dense(inputs=x, units=FLAGS.MAX_CAPTCHA * FLAGS.CHAR_SET_LEN, name='final')

    # 预测
    predictions = {
        'x_predict': tf.reshape(logits, [-1, FLAGS.MAX_CAPTCHA, FLAGS.CHAR_SET_LEN], name="x_predict")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 计算loss（对于train和valid模式）
    loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[:, 0:16], labels=labels[:, 0:16]))
    loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[:, 16:32], labels=labels[:, 16:32]))
    loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[:, 32:48], labels=labels[:, 32:48]))
    loss4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[:, 48:], labels=labels[:, 48:]))
    loss = (loss1 + loss2 + loss3 + loss4) / 4.0

    # 评估方法
    max_idx_p = tf.argmax(predictions['x_predict'], 2)
    max_idx_l = tf.argmax(tf.reshape(labels, [-1, FLAGS.MAX_CAPTCHA, FLAGS.CHAR_SET_LEN]), 2)

    correct_pred = tf.equal(max_idx_p, max_idx_l)
    batch_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 为了打印训练中的结果
    accuracy, update_op = tf.metrics.accuracy(
        labels=max_idx_p, predictions=max_idx_l, name='accuracy'
    )

    tf.summary.scalar('batch_acc', batch_acc)
    tf.summary.scalar('streaming_acc', update_op)

    # 训练配置(对于train模式)
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': (accuracy, update_op)
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# 用于解析tfrecords数据
def _parse_function(proto):
    features = {'label0': tf.FixedLenFeature([], tf.int64),
                'label1': tf.FixedLenFeature([], tf.int64),
                'label2': tf.FixedLenFeature([], tf.int64),
                'label3': tf.FixedLenFeature([], tf.int64),
                'image_encoded': tf.FixedLenFeature([], tf.string, default_value='')}

    parsed_feature = tf.parse_single_example(proto, features)
    image = tf.image.decode_image(parsed_feature['image_encoded'], channels=FLAGS.IMAGE_CHANNELS)
    image = tf.reshape(image, [FLAGS.IMAGE_HEIGHT, FLAGS.IMAGE_WIDTH, FLAGS.IMAGE_CHANNELS])
    image = tf.cast(image, dtype=tf.float32)  # 像素值需转换为float，后面送入卷积层参与计算
    image = tf.divide(tf.subtract(image, 128.0), 128.0)  # 图片标准化

    image_label0 = tf.cast(parsed_feature['label0'], tf.int32)  # 首先转为整型，再进行one hot编码
    image_label1 = tf.cast(parsed_feature['label1'], tf.int32)
    image_label2 = tf.cast(parsed_feature['label2'], tf.int32)
    image_label3 = tf.cast(parsed_feature['label3'], tf.int32)

    image_label0 = tf.one_hot(image_label0, depth=FLAGS.CHAR_SET_LEN, axis=0)  # axis=0和1是一样的效果
    image_label1 = tf.one_hot(image_label1, depth=FLAGS.CHAR_SET_LEN, axis=0)
    image_label2 = tf.one_hot(image_label2, depth=FLAGS.CHAR_SET_LEN, axis=0)
    image_label3 = tf.one_hot(image_label3, depth=FLAGS.CHAR_SET_LEN, axis=0)

    image_label = tf.concat([image_label0, image_label1, image_label2, image_label3], axis=0)

    return image, image_label


def main(unused_argv):
    # 读取训练数据集
    def train_input_fn():
        '''
        训练输入函数，返回一个batch的features和labels
        :return:
        '''
        train_dataset = tf.data.TFRecordDataset(FLAGS.train_dataset)
        train_dataset = train_dataset.map(_parse_function, num_parallel_calls=8)
        train_dataset = train_dataset.repeat(FLAGS.num_epochs)
        train_dataset = train_dataset.batch(FLAGS.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        features, labels = train_iterator.get_next()

        return features, labels

    # 读取验证数据集
    def valid_input_fn():
        '''
        验证输入函数，返回一个batch的features和labels
        :return:
        '''
        valid_dataset = tf.data.TFRecordDataset(FLAGS.valid_dataset)
        valid_dataset = valid_dataset.map(_parse_function, num_parallel_calls=8)
        # valid_dataset = valid_dataset.repeat(FLAGS.num_epochs)
        valid_dataset = valid_dataset.batch(FLAGS.batch_size)
        valid_dataset = valid_dataset.make_one_shot_iterator()
        features, labels = valid_dataset.get_next()

        return features, labels

    # run模型
    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=FLAGS.warm_model_dir,
                                        vars_to_warm_start="conv*")

    classifier_ = tf.estimator.Estimator(
        model_fn=lenet_model_fn, model_dir=FLAGS.model_dir, warm_start_from=ws
    )

    classifier_.train(input_fn=train_input_fn)
    valid_results = classifier_.evaluate(input_fn=valid_input_fn)
    print(valid_results)


if __name__ == "__main__":
    tf.app.run()