  *	2333??@2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2 '1?Z??!1?N?<?@@)'1?Z??11?N?<?@@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat Dio?????!?v??L0@)?4?8EG??1??_??-@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat <Nё\???!??Y?v/@)??A?f??1??5???+@:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::Prefetch::MapGx$(@!?|q???H@)?? ?	??1ϛ%b5?@:Preprocessing2?
[Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip ?):????!"?z?-w:@)lxz?,C??1K?$o??@:Preprocessing2?
rIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip ??j+????!¡?Kr?6@)?8EGr???1)?{?-?@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle 鷯???!L%?c>C@)H?}8g??1֘??7Q@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2 R???Q??!0F?iY@)R???Q??10F?iY@:Preprocessing2u
>Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2 ?o_???!v?,SD@) ?~?:p??1??z???@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@#??~j???!8b!)c???)#??~j???18b!)c???:Preprocessing2?
yIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat::FromTensor@n????!)?d읶??)n????1)?d읶??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?W[?????! )]????)?W[?????1 )]????:Preprocessing2?
|Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[0]::Range ݵ?|г??!(??b??)ݵ?|г??1(??b??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?k	??g??!Ƨ?????)? ?	???1it*|???:Preprocessing2F
Iterator::Model???3???!???Z????)a2U0*?s?1H?R?? ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.