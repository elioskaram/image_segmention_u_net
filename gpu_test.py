import tensorflow as tf
from datetime import datetime

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) == 0:
    print("No GPU detected. Ensure CUDA and cuDNN are installed and properly configured.")
else:
    print("TensorFlow detected GPU(s).")

# Choose size of the matrix to be used.
shapes = [(50, 50), (100, 100), (500, 500), (1000, 1000)]

def compute_operations(device, shape):
    with tf.compat.v1.Session() as session:
        with tf.device(device):
            random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
            dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
            sum_operation = tf.reduce_sum(dot_operation)

            # Time the actual runtime of the operations
            start_time = datetime.now()
            result = session.run(sum_operation)
            elapsed_time = datetime.now() - start_time

    return result, elapsed_time



if __name__ == '__main__':
    devices = ['/cpu:0', '/gpu:0']

    for device in devices:
        print("--" * 20)
        print(f"Testing on device: {device}")

        for shape in shapes:
            try:
                _, time_taken = compute_operations(device, shape)
                print(f"Input shape: {shape} using Device: {device} took: {time_taken.total_seconds():.6f} seconds")
            except RuntimeError as e:
                print(f"Failed to compute on device: {device} with shape: {shape}. Error: {str(e)}")
    print("--" * 20)
