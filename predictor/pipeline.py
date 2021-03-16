import requests
import os
import time

from metaflow import FlowSpec, step


class ImageFlow(FlowSpec):

    @step
    def start(self):
        """
        Gets images and prepares them to be sent to server
        """

        all_paths = []

        print('starting image prediction pipline')

        # wait 5 seconds for flask server to start up
        time.sleep(5)

        # get file paths for all valid images in data folder
        for file_path in os.listdir("./data"):

            # only predict jpg files
            if file_path.endswith(".jpg"):
                full_path = os.path.join("./data", file_path)
                all_paths.append(full_path)

        self.images_to_predict = all_paths
        self.next(self.get_predictions)

    @step
    def get_predictions(self):
        """
        Makes requests for predictions and saves results
        """

        results = {}

        for idx, image_path in enumerate(self.images_to_predict):

            # make request to flask server in other docker image
            resp = requests.post("http://web:5000/predict",
                                 files={"file": open(image_path, 'rb')})

            full_response = resp.json()
            results[idx] = full_response

        self.results = results
        self.next(self.end)

    @step
    def end(self):
        """
        Returns predictions
        """

        print('Getting predictions for {} images'.format(len(self.results)))

        for image_id, output in self.results.items():
            image_class = output.get('class_name')
            image_num = image_id + 1
            print('image #{} classified as: {}'.format(image_num, image_class))


if __name__ == '__main__':
    ImageFlow()
