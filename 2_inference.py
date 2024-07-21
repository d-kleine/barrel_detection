import numpy as np
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt

def calculate_angle(pt1, pt2, pt3, pt4):
    vector1 = np.array(pt2) - np.array(pt1)
    vector2 = np.array(pt4) - np.array(pt3)
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    angle = np.degrees(angle)
    return angle

def main(image_path, keypoint):
    # Load the YOLOv8 model
    model = YOLO("runs/barrel/yolov8n_custom/weights/best.pt")

    # Run inference on the image
    results = model(image_path)

    # Get the first result (assuming only one image was passed)
    result = results[0]

    # Display the results
    result_img = result.plot()  # This plots bounding boxes and labels on the image

    # Convert the image from BGR to RGB
    result_img_rgb = result_img[:, :, ::-1]

    # Display the image
    plt.imshow(result_img_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

    # Use this below code if inference fails for single image
    # # Make predictions
    # results = model.predict(image_path, save=False, conf=0.5) # adjust confidence score treshold value if necessary

    # # Visualize the results
    # for result in results:
    #     # Plot results image
    #     im_bgr = result.plot()  # BGR-order numpy array
    #     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # Convert to RGB-order PIL image

    #     # Show the image (optional, depending on your environment)
    #     im_rgb.show()

    # Retrieve and print the class names from the model
    class_names = model.model.names
    for class_index, class_name in class_names.items():
        print(f"Class {class_index}: {class_name}")

    # Extract keypoints for the barrel and the tank
    keypoints_data = result.keypoints.xy.cpu().numpy()
    kp_with_id = [[(i, x, y) for i, (x, y) in enumerate(kp) if (x, y) != (0.0, 0.0)] for kp in keypoints_data]
    print(kp_with_id)

    # Assuming the first row is for the barrel and the second for the tank
    turret_keypoints = keypoints_data[0]
    hull_keypoints = keypoints_data[1]

    # Extract the confidence scores
    confidence_scores = result.keypoints.conf.cpu().numpy()
    hull_confidences = confidence_scores[1]

    # Assign the confidence score for the top-right keypoint of the hull to a variable
    confidence_kp_hull_topright = hull_confidences[0]

    # Assign the confidence score for the bottom-right keypoint of the hull to a variable
    confidence_kp_hull_bottomright = hull_confidences[1]

    # Print the confidence scores
    print(f"Confidence score for top-right keypoint: {confidence_kp_hull_topright}")
    print(f"Confidence score for bottom-right keypoint: {confidence_kp_hull_bottomright}")

    # Extract the confidence scores for the turret keypoints from the first element of the confidence_scores list
    turret_confidences = confidence_scores[0]

    # Assign the confidence score for the intersection keypoint of the turret to a variable
    confidence_kp_turret_intersection = turret_confidences[3]

    # Assign the confidence score for the top keypoint of the turret to a variable
    confidence_kp_turret_top = turret_confidences[1]

    # Print the confidence scores
    print(f"Confidence score for intersection keypoint: {confidence_kp_turret_intersection}")
    print(f"Confidence score for top keypoint: {confidence_kp_turret_top}")

    # Define the keypoints for barrel and tank
    barrel_points = {
        "muzzle": turret_keypoints[0],
        "intersection": turret_keypoints[1] if keypoint == "turret_keypoints[1]" else turret_keypoints[3]  # Use top keypoint or intersection
    }

    tank_points = {
        "top-right": hull_keypoints[0],
        "bottom-right": hull_keypoints[1]
    }

    # Calculate the rotation angle between the barrel and the tank
    angle = calculate_angle(barrel_points["muzzle"], barrel_points["intersection"], tank_points["top-right"], tank_points["bottom-right"])

    # Print the end barrel as a vector
    print(f"Barrel vector: Barrel end (Muzzle) ({barrel_points['muzzle'][0]:.2f}, {barrel_points['muzzle'][1]:.2f}), Barrel start ({barrel_points['intersection'][0]:.2f}, {barrel_points['intersection'][1]:.2f})")

    # Print the rotation angle
    print(f"Estimated Rotation Angle: {angle:.2f} degrees")

    # Determine the direction of the barrel relative to the hull
    if angle == 0:
        print("The barrel is pointing at the front of the hull.")
    elif angle == 180 or angle == -180:
        print("The barrel is pointing at the rear of the hull.")
    elif angle > 0:
        print(f"The barrel is oriented {angle:.2f} degrees to the right of the hull.")
    else:
        print(f"The barrel is oriented {abs(angle):.2f} degrees to the left of the hull.")

    # Calculate the orientation vector components
    muzzle_x, muzzle_y = barrel_points['muzzle']
    intersection_x, intersection_y = barrel_points['intersection']
    orientation_vector_x = muzzle_x - intersection_x
    orientation_vector_y = muzzle_y - intersection_y

    # Print the orientation vector
    print(f"Barrel Orientation vector: ({orientation_vector_x:.2f}, {orientation_vector_y:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a test image.")
    parser.add_argument("image_path", type=str, help="Path to the test image.")
    parser.add_argument("keypoint", type=str, choices=["turret_keypoints[1]", "turret_keypoints[3]"], help="Choose between turret_keypoints[1] or turret_keypoints[3].")
    args = parser.parse_args()
    main(args.image_path, args.keypoint)
