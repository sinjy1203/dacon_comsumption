  *	effffL?@2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2 ???????!qI????A@)???????1qI????A@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat ??g??s??!KAh??7/@)HP?s??1??#?i ,@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat ?9#J{???!Gv??/,@)??C?l???1wB?ߤ?(@:Preprocessing2?
[Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip ?D?????!.y?˕?;@)?<,Ԛ???1???$н@:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::Prefetch::Map?\?C????!?0zS??H@)??7??d??1?[2֚@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2 ?H?}??!,???1?@)?H?}??1,???1?@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle ??m4????!?Y?QD@)??C?l??1vk&???@:Preprocessing2?
rIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip HP?s???!'??t?m4@)M??St$??1#/??x?@:Preprocessing2u
>Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2?e??a???!.e9?&F@)u????1????a@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@?!??u???!???^???)?!??u???1???^???:Preprocessing2?
yIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat::FromTensor@???S㥛?!Z?$Z???)???S㥛?1Z?$Z???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?~j?t???!?J=????)?~j?t???1?J=????:Preprocessing2?
|Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[0]::Range Dio??ɔ?!??+????)Dio??ɔ?1??+????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismvq?-??!A+??{???)ŏ1w-!?1?+?%???:Preprocessing2F
Iterator::Model7?[ A??!U*?????)?J?4a?1,?"?_ž?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.