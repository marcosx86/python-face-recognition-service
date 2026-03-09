import requests
import time

def run_test():
    print("Testing registration...")
    with open('test_face.jpg', 'wb') as f:
        # We need a valid JPG to not fail the face_recognition.load_image_file...
        # Let's generate a dummy 200x200 image using numpy and cv2
        pass

if __name__ == '__main__':
    run_test()
