# PPO agents

## 実行方法

```bash
$python -m agents.scripts.train --logdir=./logdir --config=pendulum
$python -m agents.scripts.visualize --logdir=/path/to/logdir/<time>-<config> --outdir=/path/to/outdir/
```

## 概要

<img src="./assets/graph1.png" alt="サンプル" width="1000">

<img src="./assets/graph2.png" alt="サンプル" width="300">

|ファイル|内容|
|:----|:--|
|configs.py|タスクやアルゴリズムを特定する実験configurations|
|networks.py|ニューラルネットワークモデル|
|train.py|学習セットアップを含む実行可能ファイル|
|ppo/ppo.py|PPOアルゴリズムのTensorFlowグラフ|

以下の用意されたインターフェースによって、OpenAIのgymをより効率的にアルゴリズムの実装できるように統合されている。

- **agents.tools.wrappers.ExternalProcess**はOpenAI Gym環境の外部プロセスの内側を構築してくれる環境ラッパー。`steps()`と`reset()`をアトリビュート(属性)アクセスとして呼べ、プロセスへおくられ結果を待つ。これはPythonのglobal interpreter lockによって制限されないでマルチ環境の実行を許す。
- **agents.tools.BatchEnv**は複数のOpenAI Gymの環境をバッチとして捉えれるインターフェースになっている。`step()`を行動のバッチで呼ぶと、観測、報酬、情報等のバッチが与えられる。`個々の環境が外部プロセスに存在する場合、それらは並行してステップされる(??)`
- **agents.tools.InGraphBatchEnv**はTensorFlowのグラフに統合され、`step(), reset()`がオペレーションとして作られる。現在の観測、最後の行動、報酬や端末フラグのバッチ、はtensorとして入手可能で、`tf.Variable`にsotreされる
- **agents.tools.simulate()**は`in-graph`バッチ環境のステップと強化学習を一つのオペレーションに融合し、学習ループの内部で呼べるようにする。これにより、`tf.Session`のコールの数が減らせる

## Tools

### tools.mock_environment.py

実装した。一言でいうと、まさにOpenAI Gymの型にハマった環境のモックだった。`step()`と`reset()`が用意されている。

`duration`は期間という意味で、`n[duration] = n[steps] / [episode]`という感じ。勉強になったことといえば、durationが「大体これぐらいの長さ(step)だったらいいな」に対して、stepsは「実際この長さ(stepの)だった」というもの。    
なので、モックの場合はdurations[-1]決定的で、steps[-1]は単調増加してdurations[-1]を超過すれば`done`する

### tools.mock_algorithm.py

実装した。`perform()`, `experience()`, `end_episode()`がなかなか意味深。なんとなくではあるけど、TensorBoardをみる限り

#### 学習時

<img src="./assets/end_episode.png" alt="サンプル" width="700">

<img src="./assets/experience.png" alt="サンプル" width="700">

#### 評価(eval)時

<img src="./assets/simulate.png" alt="サンプル" width="500">


----

TensorBoardの`DISTRIBUTIONS, HISTGRAM`を見るとある程度わかる。

**perform**は、行動、対数確率、モード、偏差を実行し、監視してる


いや、やっぱりよくわからん。**！！！！これらのメソッドの流れをしっかり理解する！！！！**はい

### tools.AttrDict

これは正直わかりやすい。dict()を改良したもので属性としてKeyにアクセスできて、lock機能がある。withでunlockしてからじゃないとアクセスできない。immutableというはず。なぜこんなものが必要なのかよくわからん

### tools.batch_env

複数の環境を組み合わせて、それらを一括して実行します。

環境を並行して進めるためには、環境は後で結果を受け取る代わりに、呼び出し可能なものを返すようにstepとresetの関数に対して `blocking = False`引数をサポートしなければなりません。

↑正直意味がわからん。

`blocking`って同期のことで、`blocking=False`のときは、実行せずに`ExternalProcess._recv`が帰ってくる。それをExternalProcess内で実行するか、listとしてGroupで持たせてから実行するかで意味が変わってくる←わかるかよ

↑多分だけど、`message, payload = self._conn.recv()`←親プロセスのrecvは実行されてるけど、`payload`を実行するかどうかで、`self._worker`つまり子プロセスが実行するかどうかなんじゃないだろうか…←知るか






### tools.count_weights

用途は正直イマイチわかっていない。しかしやっていることは明確で、すべての重みの数を数え上げている。

勉強になったのは、`import re`から、`.match`したものは除外するというプログラムだ。    
あと、ニューラルネットワークのモデルの定義を`with tf.name_scope`内で書いても、`x.name`には反映されてない。    
モデル名に反映させたかったら、`tf.variable_scope`を使わないといけないっぽい

### tools.in_graph_batch_env

1. 複数の`tools.ExternalProcess`
2. それをまとめる`tools.BatchEnv`
3. in-Graph化する`tools.InGraphBatchEnv`

という流れです。

`tools.in_graph_env`とほとんど内容が変わらない。リストで扱ってるわけじゃなくて、`tools.BatchEnv`のインスタンスが渡されているから`__len__()`でBatchサイズを指定する必要がある

正直なんでこんなことをするかわからないけど、

```python
    with tf.control_dependencies([
      tf.scatter_update(self._observ, indices, observ),
      tf.scatter_update(self._reward, indices, reward),
      tf.scatter_update(self._done, indices, done)]):
      return tf.identity(observ)
```      

をしている。`tf.scatter_update`と`tf.assign`は同値なはず。それはあくまで`indices`を与えていない場合。与えればもっと自由度のある書き方ができる。おそらく、どっかの`reset`時にもっと自由度のある書き方をしてるんじゃないだろうか。

### tools.in_graph_env

`InGrapEnv`を実装した。このクラスはこの`agents`ライブラリで全く使われていない。念のため、テストを書いた。

このクラスは、`tf.Variable`で観測、報酬、端末フラグを管理しており、端的に言えば`gym.Env`のラッパーである。   
`simulate()`が環境の更新関数なのかな。`gym.Env.step`に似ている。違いは、`simulate()`はTensorFlow仕様になっているから、`assign`を使っている。`self._step`というクラス変数があるけど、これは`global_step`に近い変数で、このインスタンスが何step目かを管理している

ちょっとわからないのだけど、`self.step`は`self.reset()`しても`0`が代入されていない。

勉強になったのは、Attributorの使い方である。これは、`__getattr__(self, name)`からGymのenvに直接アクセスしてる。   
他には管理している変数の名前空間は`environment`であるのに対して、`simulate()`の中では、`environment/simulate`という名前空間を使っている。    
型がいっぱいあって、担保しきれない場合は、`raise NotImplementError()`と呼ぶ。

```python
      observ, reward, done = tf.py_func(
          lambda a: self._env.step(a)[:3], [action],
          [observ_dtype, tf.float32, tf.bool], name='step')
```

この書き方は普通に勉強になるかなあ。

```python
      if action.dtype in (tf.float16, tf.float32, tf.float64):
        action = tf.check_numerics(action, 'action')
```

みたいに、どういう値がAgent側で担保しない場合は、環境が担保する。

### tools.loop

主に強化学習の`loop`部分に当たる。

しかし、あまりにも分かりづらい。難しい。

だから、テストから理解していこうかなって思う

`test_not_done`から。

`Loop()`のインスタンスに`.add_phase()`というインスタンスメソッドで登録していく。これは勉強になる。

envの終了条件(`done`)を`(step + 1) % 2 == 0`のときである。一般的に、envの終了条件はPoleが倒れたときとかである。

テストでは、
```python
In [28]: [i for i in range(9) if (i + 1 )%2 == 0]
Out[28]: [1, 3, 5, 7]
```
のときである。

そしてレポート(scoreを返すとき)ときは3回に1回なので、

```python
In [32]: [i for i in range(9) if (i+1) % 3 == 0]
Out[32]: [2, 5, 8]
```

のときである。

```python
# Step:   0 1 2 3 4 5 6 7 8
# Score:  0 1 2 3 4 5 6 7 8
# Done:     x   x   x   x
# Report:     x     x     x
```

は正しいことがわかる。

`add_phase`の引数である`steps`が何を意味してるかなんとなくわかった。phaseを何回連続、scoreを返すかというものっぽい。

`test_phases_feed`を見たらわかるけど、1を1回、2(score=1)を3回連続(steps=3)、3(score=3)を2回(steps=2)となっている

`report_every`はその何回目の`steps`でscoreを`yield`するかということ。


### tools.nested

### tools.simulate

- score: それぞれのエージェントのrewardの総和
- length: それぞれのエージェントのsteps数



**勉強できたこと**

なぜ、`_define_begin_episode(agent_indices)`とかあるのにクラスではなく`simulate`が関数なのか気になった。それは、インスタンス変数とかを所持する必要がないからだと思う。

必要なのは、`done, score, summary`だけ。だから、それだけ返すようなものを持っておけば良い

```python
core_summary = tf.cond(
      tf.logical_and(log, tf.cast(mean_score.count, tf.bool)),
      lambda: tf.summary.scalar('mean_score', mean_score.clear()),
      str
      
In [9]: tf.where([True, False, True, True])
Out[9]:
<tf.Tensor: id=12, shape=(3, 1), dtype=int64, numpy=
array([[0],
       [2],
       [3]])>


In [10]: tf.where([True, False, True, True])[:, 0]
Out[10]: <tf.Tensor: id=19, shape=(3,), dtype=int64, numpy=array([0, 2, 3])>
In [14]: tf.where([True, False, True, True])
Out[14]:
<tf.Tensor: id=36, shape=(3, 1), dtype=int64, numpy=
array([[0],
       [2],
       [3]])>


In [15]: tf.where([True, False, True, True]).shape.ndims
Out[15]: 2
```

一瞬わけわからんけど、`tf.logical_and`は`tf.cond`に併用できる。あと、`.count`みたいな初期値状態に対してbool値と見立ててれる。

けど、よくよく考えてみれば、下みたいな方が単純なような気がする

```python
if log and bool(mean_score.count):
	return tf.summary....
else:
	return str	
```




### tools.streaming_mean

`tools.simulate`の中で使われる常にある値を追いかけながら、カウントも同時に確認するので、平均を求めることができる。

`submit`で`assign_add`の`tf.group`を返す

今回、`agents`

### tools.wrappers

テストでの使われ方。

```python
env = tools.ExternalProcess(original_env)
env.reset()
env.step(...)
env.step(...)
env.close()
```

一見、`gym.Env`との違いがわからない。

基本的に

1. 親プロセス、子プロセスを作る
2. 子プロセスに`self._worker`の仕事をさせる。ここできちんと理解しないといけないのが、子プロセスは`KeyboardInterupt`のような例外が起こらなければ、一生回り続けているということ。
3. `reset(), step()`のようなインターフェース(`close()`は除く)をユーザが呼び出す
    1. reset()やstep()内部でself.call()をする。
        2. self.call()の中ではメッセージとname(`reset`, `step`, ...)を子プロセスにsendする
    2. 子プロセスがそれをrecvして、constructor(i.g. `gym.Env`)からgetattrしてきて、再びsendする
    3. self.call()はcallableなreceiveを返す。ここで子プロセスからの情報をrecvして、齟齬がないようにする。
    


勉強になったことで、まずはテストから。   
`MockEnvironment`を使っている。呼び出し可能な状態の環境を渡すために、`functions.partial`ですべての引数を渡している。こうすることによって、`gym.make(...)`と同じようなインスタンス(ライク)なものができる。モックすごい。

あと、`atexit.Register`で`self.close`をしている。`atexit.Register`は、正常終了したときに呼び出したい関数を呼んでくれるRegister。書いて思ったけどGoの`defer`に似てるかもしれない

クラスで状態フラグを定数として管理をしている。このフラグは`_CALL, _CLOSE, _ACCESS, _EXCEPTION, _RESULT`。あ、クラス変数か。

あと、子プロセスに永遠と仕事させて、フラグでメタ的に`send, recv`に情報を与えてるのが面白かった。


## scripts.utility

アルゴリズムを実装するためのutility.

### define_simulation_graph(batch_env, algo_cls, config):

### define_batch_env(constructor, num_agents, env_processes)

### define_saver()