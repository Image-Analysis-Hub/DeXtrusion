"""
Compatibility layer for Keras 2.x and 3.x
Handles API differences between TensorFlow versions.
"""

import tensorflow as tf

# Check Keras version by trying to detect Keras 3 specific features
def _is_keras_3():
    """Detect if we're using Keras 3.x"""
    try:
        # Keras 3 moved to a different package structure
        # Try to import from the new location
        import keras
        if hasattr(keras, 'version') and callable(keras.version):
            version = keras.version()
            return int(version.split('.')[0]) >= 3
    except (ImportError, AttributeError):
        pass
    
    # Alternative check: TensorFlow 2.16+ bundles Keras 3
    tf_version = tf.__version__.split('.')
    tf_major = int(tf_version[0])
    tf_minor = int(tf_version[1].split('-')[0].split('rc')[0])
    
    # TF 2.16+ uses Keras 3 by default
    if tf_major > 2 or (tf_major == 2 and tf_minor >= 16):
        return True
    
    # Check if backend.function exists (removed in Keras 3)
    try:
        from tensorflow.keras.backend import function
        return False  # If function exists, it's Keras 2
    except ImportError:
        return True  # If function doesn't exist, it's Keras 3
    
    return False

KERAS_3 = _is_keras_3()

def get_keras_function():
    """Get the function builder compatible with current Keras version"""
    if KERAS_3:
        # Keras 3.x: Use tf.keras.backend.function or Model approach
        # The 'function' API changed significantly
        # Return a wrapper that mimics the old behavior
        def keras_function_wrapper(inputs, outputs):
            """Wrapper for Keras 3 compatibility"""
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return lambda x: model(x, training=False)
        return keras_function_wrapper
    else:
        # Keras 2.x: Use the traditional backend.function
        from tensorflow.keras.backend import function
        return function

def clear_session():
    """Clear Keras session - compatible with all versions"""
    tf.keras.backend.clear_session()

def init_tf_session(verbose=True):
    """Initialize TensorFlow session - compatible with TF 2.x"""
    if verbose:
        print("Tensorflow with Cuda: " + str(tf.test.is_built_with_cuda()))
        print("Tensorflow version: " + str(tf.__version__))
        
        # Get Keras version safely
        keras_version = "Unknown"
        try:
            import keras
            if hasattr(keras, 'version') and callable(keras.version):
                keras_version = keras.version()
        except ImportError:
            keras_version = f"Bundled with TF (Keras {'3' if KERAS_3 else '2'})"
        
        print("Keras version: " + keras_version)
        print("Using Keras 3 API: " + str(KERAS_3))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Modern TF 2.x doesn't need session management like TF 1.x
    # But we can configure GPU memory growth if needed
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            if verbose:
                print(f"GPU configuration: {e}")

def get_metric_name(base_name='accuracy'):
    """Get correct metric name for current Keras version"""
    # Keras 3.x uses full names like 'accuracy'
    # Keras 2.x used abbreviations like 'acc'
    if KERAS_3:
        return base_name
    else:
        # Map full names to Keras 2.x abbreviations
        mapping = {
            'accuracy': 'acc',
            'val_accuracy': 'val_acc'
        }
        return mapping.get(base_name, base_name)
    