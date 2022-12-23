import tensorflow as tf
#エントロピー計算
def cal_entropy(tensor):
    #画像のエントロピーを計算する
    """
    tensor: 画像の変数テンソル
    """
    #テンソルを平たんにする
    flatten_tensor = tf.reshape(tensor,[-1])
    size = tf.cast(tf.shape(flatten_tensor), tf.float32)
    bit = 0.0000001
    #画素値ごとの出現確立を求める
    where = tf.where(tf.math.equal(tf.cast(flatten_tensor, tf.int32), 0))
    num = tf.reduce_sum(tf.gather_nd(flatten_tensor, where)/(tf.gather_nd(flatten_tensor, where)+bit))
    p_0 = num / size
    fn = lambda:-p_0*0
    bn = lambda:-p_0 * tf.log(p_0)
    result =tf.cond(tf.squeeze(tf.math.equal(p_0,0)),fn,bn)
    entropy = result
    for i in range(1, 256):
        #画素値iの位置を求める
        where = tf.where(tf.math.equal(tf.cast(flatten_tensor,tf.int32), i))
        #テンソルに含まれる画素値iの個数を計算する
        num = tf.cast(tf.reduce_sum(tf.gather_nd(flatten_tensor, where)/(tf.gather_nd(flatten_tensor, where)+bit)),tf.float32)
        #画素値iの出現確立
        p_i = num / size
        #エントロピーの加算
        fn = lambda:-p_i*0
        bn = lambda:-p_i * tf.log(p_i)
        result = tf.cond(tf.squeeze(tf.math.equal(p_i,0)),fn,bn)
        entropy = entropy + result
    
    return entropy