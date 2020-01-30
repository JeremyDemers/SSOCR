import cv2
import numpy as np
import argparse

# (top-right,bottom-right,bottom,bottom-left,left-top,top,middle)
DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 0, 0, 1, 1, 1): 9,
    (0, 0, 0, 0, 0, 1, 1): '-'
}
H_W_Ratio = 1.9
THRESHOLD = 35
arc_tan_theta = 6.0  # Digital tube tilt angle

parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='path to image')
parser.add_argument('-s', '--show_image', action='store_const', const=True, help='whether to show image')


def load_image(path, show=False):
    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    if show:
        cv2.imshow('gray_img', gray_img)
        cv2.imshow('blurred_img', blurred)
    return blurred, gray_img


def preprocess(img, threshold, show=False, kernel_size=(5, 5)):
    # Histogram local equalization
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    img = clahe.apply(img)
    # Adaptive threshold binarization
    dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)
    # Closed operation open operation
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)

    if show:
        cv2.imshow('equlizeHist', img)
        cv2.imshow('threshold', dst)
    return dst


def helper_extract(one_d_array, threshold=20):
    res = []
    flag = 0
    temp = 0
    for i in range(len(one_d_array)):
        if one_d_array[i] < 12 * 255:
            if flag > threshold:
                start = i - flag
                end = i
                temp = end
                if end - start > 20:
                    res.append((start, end))
            flag = 0
        else:
            flag += 1

    else:
        if flag > threshold:
            start = temp
            end = len(one_d_array)
            if end - start > 50:
                res.append((start, end))
    return res


def find_digits_positions(img, reserved_threshold=20):
    digits_positions = []
    img_array = np.sum(img, axis=0)
    horizon_position = helper_extract(img_array, threshold=reserved_threshold)
    img_array = np.sum(img, axis=1)
    vertical_position = helper_extract(img_array, threshold=reserved_threshold * 4)

    # make vertical_position has only one element
    if len(vertical_position) > 1:
        vertical_position = [(vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])]
    for h in horizon_position:
        for v in vertical_position:
            digits_positions.append(list(zip(h, v)))
    assert len(digits_positions) > 0, "Failed to find digits's positions"

    return digits_positions


def recognize_digits_line_method(digits_positions, output_img, input_img):
    output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)
    digits = []
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_w = max(1, int(h / H_W_Ratio))

        # Eliminate extraneous symbol interference
        if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
            continue

        if w < suppose_w / 2:
            x0 = max(x0 + w - suppose_w, 0)
            roi = input_img[y0:y1, x0:x1]

        center_y = h // 2
        quarter_y_1 = h // 4
        quarter_y_3 = quarter_y_1 * 3
        center_x = w // 2
        line_width = 5  # line's width
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        small_delta = int(h / arc_tan_theta) // 4
        segments = [
            ((w - 2 * width, quarter_y_1 - line_width), (w, quarter_y_1 + line_width)),
            ((w - 2 * width, quarter_y_3 - line_width), (w, quarter_y_3 + line_width)),
            ((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
            ((0, quarter_y_3 - line_width), (2 * width, quarter_y_3 + line_width)),
            ((0, quarter_y_1 - line_width), (2 * width, quarter_y_1 + line_width)),
            ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
            ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
        ]
        on = [0] * len(segments)

        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            if total / float(area) > 0.25:
                on[i] = 1
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '*'

        digits.append(digit)

        # decimal point recognition
        if cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9. / 16 * width * width) > 0.65:
            digits.append('.')
            cv2.rectangle(output_img,
                          (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4)),
                          (x1, y1), (0, 128, 0), 2)
            cv2.putText(output_img, 'dot',
                        (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)

        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 200, 0), 2)
        cv2.putText(output_img, str(digit), (x0 + 3, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2)
    return digits, output_img


def main():
    args = parser.parse_args()
    blurred, gray_img = load_image(args.image_path, show=args.show_image)
    output = blurred
    dst = preprocess(blurred, THRESHOLD, show=args.show_image)
    digits_positions = find_digits_positions(dst)
    digits, output_img = recognize_digits_line_method(digits_positions, output, dst)
    if args.show_image:
        cv2.imshow('output', output_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    print(digits)


if __name__ == '__main__':
    main()
