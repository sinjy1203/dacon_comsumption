  *	33333?@2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2 ?$??C??!?????uB@)?$??C??1?????uB@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat -???????!?s5?'+1@)Z??ڊ???1?I[1?</@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat ??St$???!?y?u?l+@)?N@a???1ݛ?I(@:Preprocessing2?
[Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip ??	h"l??!!??nʬ8@)a2U0*???1??,[?!@:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::Prefetch::Map?o_???!?;?$?CI@)?A`??"??1?Y,o??@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle Gx$(??!??9??D@)M??St$??1?E??S@:Preprocessing2?
rIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip ???_vO??! 9??g?6@)/?$???1?ḛ2?@:Preprocessing2u
>Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2?g??s???!`????QF@)?E???Ԩ?1?}?Վ@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2 V????_??!??3?l@)V????_??1??3?l@:Preprocessing2?
yIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat::FromTensor@u????!8??̪???)u????18??̪???:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@6?;Nё??!?|h????)6?;Nё??1?|h????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?A`??"??!?Y,o????)?A`??"??1?Y,o????:Preprocessing2?
|Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[0]::Range ?:pΈ??!$?H?6??)?:pΈ??1$?H?6??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismm???{???!`???l??)F%u?{?1Ls]]"x??:Preprocessing2F
Iterator::Model/n????!??||?J??)?J?4a?1?5?_?޽?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.