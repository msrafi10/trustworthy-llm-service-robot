import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import joblib
import os
import numpy as np


class IntentNode(Node):

    def __init__(self):
        super().__init__('intent_node')

        base_path = os.path.dirname(__file__)

        # Load model
        self.model = joblib.load(os.path.join(base_path, "intent_model.pkl"))
        self.vectorizer = joblib.load(os.path.join(base_path, "vectorizer.pkl"))

        # Subscriber
        self.subscription = self.create_subscription(
            String,
            'user_command',
            self.listener_callback,
            10)

        # Publisher
        self.publisher = self.create_publisher(
            String,
            'detected_intent',
            10)

        self.get_logger().info("Intent Safety Node Started ✅")


    def listener_callback(self, msg):

        text = msg.data
        text_vec = self.vectorizer.transform([text])

        prediction = self.model.predict(text_vec)[0]
        confidence = np.max(self.model.predict_proba(text_vec))

        output_msg = String()
        output_msg.data = f"{prediction} | confidence: {confidence:.2f}"

        self.publisher.publish(output_msg)

        self.get_logger().info(
            f"Input: {text} → Intent: {prediction} (Conf: {confidence:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = IntentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
