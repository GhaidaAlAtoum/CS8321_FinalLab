import tensorflow as tf

# Reference: https://gist.github.com/shawnbutts/3906915
def bytes_to(bytes, to, bsize=1024):
    """convert bytes to megabytes, etc.
       sample code:
           print('mb= ' + str(bytes_to(314575262000000, 'm')))
       sample output: 
           mb= 300002347.946
    """

    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    for i in range(a[to]):
        r = r / bsize

    return(r)

def print_gpu_usage():
    print("GPU Peak Memory: ", bytes_to(tf.config.experimental.get_memory_info('GPU:0')['peak'], 'g'), 'G')
    print("GPU Current Memory Usage: ", bytes_to(tf.config.experimental.get_memory_usage('GPU:0'), 'g'), 'G')