import numpy as np
import pandas as pd
import math
import cv2
import os
from PIL import Image

DATASET = 'INDEREBv2'

PATH = os.path.dirname(os.path.abspath(__file__))
BLOOD_VESSELS_PATH = os.path.join(PATH, DATASET, 'resources', 'blood_vessels', 'test', 'pred_only')
OPTIC_DISK_PATH = os.path.join(PATH, DATASET, 'resources', 'optic_disk', 'test', 'pred_only')

BLOOD_VESSELS_PATH = os.path.join(PATH, DATASET, 'resources', 'blood_vessels', 'images', 'GT')
OPTIC_DISK_PATH = os.path.join(PATH, DATASET, 'resources', 'optic_disk', 'images', 'GT')

IMAGES_PATH = os.path.join(PATH, DATASET, 'resources', 'blood_vessels', 'images', 'fundusimages')
RESULTS_PATH = os.path.join(PATH, DATASET, 'avr_results')
VEIN_COLOR = (255, 0, 0)
ARTERY_COLOR = (0, 0, 255)
RADIUS = [1.5, 2.2]
RADIUS = list(np.linspace(RADIUS[0], RADIUS[1], int(1+(RADIUS[1]-RADIUS[0])*10)))

def get_file_names(directory, ends_with='.png'):
    _list_ = [file for file in os.listdir(directory) if file.endswith(ends_with)]
    _list_.sort()
    return _list_


def optic_disc_center(img, channel=1, threshold=32):
    # keep red channel only
    img[:, :, channel] = 0

    # convert image to grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray_image, threshold, 255, 0)

    # calculate moments of binary image
    moments = cv2.moments(thresh)

    # calculate x,y coordinate of center
    c_x = int(moments["m10"] / moments["m00"])
    c_y = int(moments["m01"] / moments["m00"])
    return c_x, c_y


def draw_lines(points, img, color=(255, 255, 255), thickness=2):
    for point in points:
        # Start coordinate, here (0, 0)
        # represents the top left corner of image
        start_point = (point[0][0], point[0][1])

        # End coordinate, here (250, 250)
        # represents the bottom right corner of image
        end_point = (point[1][0], point[1][1])

        # Using cv2.line() method
        # Draw a diagonal green line with thickness of 9 px
        img = cv2.line(img, start_point, end_point, color, thickness)

    return img


def draw_text(points, img, prefix='label', color=(255, 255, 255), thickness=1, scale=0.25):
    i = 0
    for point in points:
        # Using cv2.line() method
        # Draw a diagonal green line with thickness of 9 px
        img = cv2.putText(img, prefix+'_'+str(i), (point[0][0], point[0][1]), cv2.FONT_HERSHEY_SIMPLEX, scale,
                          color, thickness, 2)
        i += 1

    return img


def get_contour_distances(img, channel, threshold=32):
    # create zero matrix
    zero_img_matrix = np.zeros(img.shape[0:2], dtype='uint8')

    # split colors
    b, g, r = cv2.split(img)

    # would like to construct a 3 channel image with only 1 channel filled
    b = cv2.merge([b, zero_img_matrix, zero_img_matrix])
    g = cv2.merge([zero_img_matrix, g, zero_img_matrix])
    r = cv2.merge([zero_img_matrix, zero_img_matrix, r])

    # pick one channel to find the contours
    img_by_channel = [b, g, r]
    contours_img = img_by_channel[channel]

    # find contours
    gray_image = cv2.cvtColor(contours_img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 1)

    # create a collection of points and their distances where distance between them in the maximum
    points = []
    distances = []
    for contour in contours:
        # reshape contour
        a = np.reshape(contour, (contour.shape[0], 2))

        # compute matrix distance between all points
        dist_matrix = np.linalg.norm(a - a[:, None], axis=-1)
        if dist_matrix.shape[0] > 1:
            # extract the max distance
            max_dist = np.amax(dist_matrix)
            distances.append(max_dist)

            # get which points produce the max distance
            positions = np.where(dist_matrix == np.amax(max_dist))
            _aux = a[np.array(positions)[0]]
            points.append([_aux[0], _aux[1]])

    return points, distances


def avr_calculation(_df, radius):
    scalars = {'artery': 0.88,
               'vein': 0.95}
    cre = {'artery': [],
           'vein': []}
    df_a = pd.DataFrame()
    df_v = pd.DataFrame()
    dfs = {'artery': df_a,
           'vein': df_v}
    for category in dfs.keys():
        df_temp = pd.DataFrame()
        for r in radius:
            df_slice = _df[(_df['category'] == category) & (_df['radius'] == r)].copy()
            df_slice.sort_values(by=['distance'], ascending=False, inplace=True)
            df_slice.reset_index(inplace=True, drop=True)
            df_temp = pd.concat([df_temp, df_slice.loc[0:3, :]])
            df_temp.sort_values(by=['distance'], ascending=False, inplace=True)
            df_temp.reset_index(inplace=True, drop=True)
            dist = df_temp['distance'].tolist()
            keep_going = True
            while keep_going:
                new_dist = []
                for i in range(int(len(dist)/2)):
                    w = scalars[category] * math.sqrt((dist[i]**2) + (dist[len(dist)-i-1]**2))
                    new_dist.append(w)
                dist = new_dist.copy()
                if len(dist) == 1:
                    keep_going = False
            cre[category].append(dist[0])
    #avr = cre['artery']/cre['vein']
    avr = np.mean(np.divide(np.asarray(cre['artery']), np.asarray(cre['vein'])))
    #avrs = np.divide(np.asarray(cre['artery']), np.asarray(cre['vein']))
    #avrs = np.delete(avrs, np.argmax(avrs))
    #avrs = np.delete(avrs, np.argmin(avrs))
    #return np.mean(avrs)
    return avr


def calculate_avr(av_image, od_labels, av_labels, file, radius):
    df = pd.DataFrame()
    c_x, c_y = optic_disc_center(od_labels, channel=1)
    p, d = get_contour_distances(od_labels, channel=2)

    # center coordinates
    center_coordinates = (c_x, c_y)

    # radius of circle
    circle_radius = int((d[0] / 2) * radius)

    # color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 2
    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    image = cv2.circle(od_labels, center_coordinates, circle_radius, color, thickness)

    # create a matrix with a circle only
    img_circle = cv2.circle(np.zeros(od_labels.shape, dtype='uint8'), center_coordinates, circle_radius,
                            color, thickness)

    zero_img_matrix = np.zeros(av_labels.shape[0:2], dtype='uint8')
    # split colors
    b, g, r = cv2.split(av_labels)

    # would like to construct a 3 channel image with only 1 channel filled
    b = cv2.merge([b, zero_img_matrix, zero_img_matrix])
    g = cv2.merge([zero_img_matrix, g, zero_img_matrix])
    r = cv2.merge([zero_img_matrix, zero_img_matrix, r])

    # intersect circle, veins and arteries
    b_slices = np.multiply(img_circle / 255, b)
    b_slices = b_slices.astype(np.uint8).copy()
    b_slices = cv2.merge([b_slices[:, :, 0], b_slices[:, :, 0], b_slices[:, :, 0]])
    r_slices = np.multiply(img_circle / 255, r)
    r_slices = r_slices.astype(np.uint8).copy()
    r_slices = cv2.merge([r_slices[:, :, 2], r_slices[:, :, 2], r_slices[:, :, 2]])

    # draw chosen circles
    image = cv2.circle(av_image, center_coordinates, circle_radius, (255, 255, 255), 1)

    p, d = get_contour_distances(b_slices, channel=1)
    res_image = draw_lines(p, av_image, color=VEIN_COLOR)
    category = 'vein'
    df_aux = pd.DataFrame({'file': [file] * len(d),
                           'category': [category] * len(d),
                           'radius': [radius] * len(d),
                           'points': p,
                           'distance': d})
    df = pd.concat([df, df_aux])

    p, d = get_contour_distances(r_slices, channel=1)
    res_image = draw_lines(p, res_image, color=ARTERY_COLOR)
    category = 'artery'
    df_aux = pd.DataFrame({'file': [file] * len(d),
                           'category': [category] * len(d),
                           'radius': [radius] * len(d),
                           'points': p,
                           'distance': d})
    df = pd.concat([df, df_aux])
    return res_image, df


if __name__ == '__main__':
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    radius = RADIUS[0]
    av_files = get_file_names(BLOOD_VESSELS_PATH, '.png')
    od_files = get_file_names(OPTIC_DISK_PATH, '.png')
    df_results = pd.DataFrame()

    for file in od_files:
        print('Processing {}'.format(file))
        image = Image.open(os.path.join(IMAGES_PATH, file))
        img_original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res_image = img_original.copy()
        for radius in RADIUS:
            # The function cv2.imread() is used to read an image. (BGR)
            image = Image.open(os.path.join(OPTIC_DISK_PATH, file))
            img_od = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = Image.open(os.path.join(BLOOD_VESSELS_PATH, file))
            img_av = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result_image, _df = calculate_avr(res_image, img_od, img_av, file, radius)
            df_results = pd.concat([df_results, _df])
        cv2.imwrite(os.path.join(RESULTS_PATH, file), result_image)
    df_results.reset_index(inplace=True, drop=True)
    df_results.loc[:, 'points'] = df_results['points'].apply(lambda x: [x[0].tolist(), x[1].tolist()])
    df_results.to_csv(os.path.join(RESULTS_PATH, 'raw_results.csv'), index=False)
    df_avr = df_results.groupby(['file'], as_index=False).apply(lambda x: avr_calculation(x, radius=RADIUS))
    df_avr.columns = ['file', 'avr']
    df_avr.to_csv(os.path.join(RESULTS_PATH, 'avr_results.csv'), index=False)
