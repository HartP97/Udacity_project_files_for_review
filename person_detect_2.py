%%writefile person_detect.py

"""SECOND OPTION FOR PYTHON SCRIPT"""

"""
!!! NOTE !!!
This is an alternative version to the one above, as we have used it in earlier lessons
Someone how I couldn't make it work that way, so the following solution tries to stay
closer to the before given TO-DO.
"""

import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    """Class that handles everything regarding queues."""

    def __init__(self):

        self.queues = []

    def add_queue(self, points):
        # Throw/Raise error if no points given
        if points is None:
            raise TypeError("No points given to queue")
        # Points != None -> add the points to queue
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max = q
            frame = image[y_min:y_max, x_min:x_max]
            yield frame

    def check_coords(self, coords):
        d = {k + 1: 0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0] > q[0] and coord[2] < q[2]:
                    d[i + 1] += 1
        return d

    
class PersonDetect:
    """Class that handles everything regarding detected people."""

    # init class attributes
    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Did you enter the correct model-path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        # load model that got passed when class was initialized
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        """
        Predicts a person's location
        - frame needs to be passed
        - after inference, returns outputs and image
        """

        input_name = self.input_name

        input_img = self.preprocess_input(image)
        input_dict = {input_name: input_img}

        infer_request_handle = self.net.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            outputs = infer_request_handle.outputs[self.output_name]

        return outputs, image

    def draw_outputs(self, coords, image):
        """
        Draws predicted location of person onto input-image
            - returns image including bounding boxes, amount of people in frame and bounding boxes that were above
            confidence threshold
        """

        people_count = 0
        detected = []
        for obj in coords[0][0]:
            if obj[2] > self.threshold:
                x_min = int(obj[3] * initial_w)
                y_min = int(obj[4] * initial_h)
                x_max = int(obj[5] * initial_w)
                y_max = int(obj[6] * initial_h)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 55, 255), 1)
                people_count += 1

                detected.append(obj)

        return image, people_count, detected

    def preprocess_outputs(self, outputs):

        """ pre-processing outputs to return the preprocessed dict """

        output_dict = {}
        for output in outputs:
            output_name = self.output_name
            output_img = output
            output_dict[output_name] = output_img

        return output_dict
        return output

    def preprocess_input(self, image):

        input_img = image

        n, c, h, w = self.input_shape

        input_img = cv2.resize(input_img, (w, h), interpolation=cv2.INTER_AREA)

        # Reshape img from HWC to CHW
        input_img = input_img.transpose((2, 0, 1))
        input_img = input_img.reshape((n, c, h, w))

        return input_img

def main(args):
    model = args.model
    device = args.device
    video_file = args.video
    max_people = args.max_people
    threshold = args.threshold
    output_path = args.output_path

    start_model_load_time = time.time()
    pd = PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue = Queue()

    try:
        queue_param = np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("Encountered an error loading queue param file!")

    try:
        cap = cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: " + video_file)
    except Exception as e:
        print("Something went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
                                (initial_w, initial_h), True)

    counter = 0
    start_inference_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1

            coords, image = pd.predict(frame)
            num_people = queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text = ""
            y_pixel = 25

            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text = ""
                y_pixel += 40
            out_video.write(image)

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time) + '\n')
            f.write(str(fps) + '\n')
            f.write(str(total_model_load_time) + '\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)

    args = parser.parse_args()

    main(args)
