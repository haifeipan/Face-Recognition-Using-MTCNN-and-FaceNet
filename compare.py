
#用来计算两张人脸图像之间的距离矩阵。需要输入的参数：
#预训练模型 图片1  图片220170512-110547 1.png 2.png

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
# import detect_face
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def main(args):
    # image_files = ['./validation_160/1/1.png','./validation_160/1/2.png']
    # image_files = ['C:/validation_160/1/04.png','C:/validation_160/1/05.png']

    filePath = 'C:/validation_160'
    file_list = os.listdir(filePath)
    a = []
    for i in range(0, 200):
        a.append(filePath + '/' + str(i + 1))

    d = []
    for i in range(0, 200):
        b = a[i]
        c = os.listdir(b)
        d.append([b + '/' + c[0], b + '/' + c[1]])
    image_files = []
    for i in range(0, 200):
        for j in range(0, 2):
            image_files.append(d[i][j])

    # image_files = ['1.png','2.png']

    # images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)

    # images = load_and_align_data(image_files, args.image_size, args.margin, args.gpu_memory_fraction)

    # plt.figure()
    # plt.imshow(images[1,:])
    # plt.show()
    # print('askhnauisd')

    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


            images = load_and_align_data(image_files, args.image_size, args.margin, args.gpu_memory_fraction)

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

            # nrof_images = len(args.image_files)
            nrof_images = len(image_files)
            # print('Images:')
            # for j in range(nrof_images):
            #     # print('%1d: %s' % (i, args.image_files[i]))
            #     print('%1d: %s' % (j, image_files[j]))
            # print('')

            # Print distance matrix
            print('Distance matrix')

            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            dist_array = np.zeros([400, 400])
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    dist_array[i][j] = dist
                    print('  %1.4f  ' % dist, end='')
                print('')

            # dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
            # print(dist)
            # df = pd.DataFrame(dist_array)
            # df.to_csv('similarity.csv')

            lis = []
            for i in range(0, 400, 2):
                lis.append(dist_array[i][i+1])
            #
            # a = sum(pd.Series(lis[0:100]) < 1)
            # b = sum(pd.Series(lis[100:200]) > 1)
            # accuracy1 = a / 100
            # accuracy2 = b / 100
            # accuracy = (a + b) / 200
            # print('accuracy1-100：%.4f' % accuracy1)
            # print('accuracy101-200：%.4f' % accuracy2)
            # print('accuracy_all：%.4f' % accuracy)
            #
            d = {'similarity': lis}
            df = pd.DataFrame(data=d)
            df.to_csv('compare.csv')

            s = df.similarity
            judge = 1
            a = sum(s[0:100] < judge)
            b = sum(s[100:200] > judge)
            accuracy1 = a / 100
            accuracy2 = b / 100
            accuracy = (a + b) / 200
            print(accuracy1)
            print(accuracy2)
            print(accuracy)

# image_paths
# image_size
# margin
# gpu_memory_fraction
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    # print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            # pnet, rnet, net = detect_face.create_mtcnn(sess, None)

    tmp_image_paths = image_paths.copy()
    img_list = []
    for image in tmp_image_paths:
        # img = misc.imread(os.path.expanduser(image), mode='RGB')
        img = misc.imread(image, mode='RGB')

        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        # bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        # if len(bounding_boxes) < 1:
        #   image_paths.remove(image)
        #   print("can't detect face, remove ", image)
        #   continue
        # det = np.squeeze(bounding_boxes[0,0:4])  #去掉了最后一个元素？
        # bb = np.zeros(4, dtype=np.int32)
        # # np.maximum：(X, Y, out=None) ，X 与 Y 逐位比较取其大者；相当于矩阵个元素比较
        # bb[0] = np.maximum(det[0]-margin/2, 0)#margin：人脸的宽和高？默认为44
        # bb[1] = np.maximum(det[1]-margin/2, 0)
        # bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        # bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        # cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

        cropped = img
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images
    # return img

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    # parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    #这是一个从外部输入参数的代码。
    main(parse_arguments(sys.argv[1:]))
