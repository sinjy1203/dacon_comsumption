  *	???????@2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2 ;M?O??!??<?֑>@);M?O??1??<?֑>@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat _?L???!@X?`0@)?E??????1a)?kC?-@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat ?u?????!??R?)/@)*:??H??1?[?	?+@:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::Prefetch::Mapi o????!c?!n??H@)jM????1?b??yE!@:Preprocessing2?
[Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip x$(~???!??9??:@)?Fx$??1?%?*??@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle ???????!u??e?B@)???<,Ժ?1!?3??@:Preprocessing2?
rIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip ?S㥛???!??/?5@)??4?8E??1
?w??@:Preprocessing2u
>Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2?z?G???!?d??_4D@)v??????1???(U@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2 HP?s??!???A$?@)HP?s??1???A$?@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??_?L??!?z???@)??_?L??1?z???@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@Ǻ????!??]J?M??)Ǻ????1??]J?M??:Preprocessing2?
yIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat::FromTensor@S?!?uq??!????;??)S?!?uq??1????;??:Preprocessing2?
|Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[0]::Range ??ZӼ???!??jr??)??ZӼ???1??jr??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismx$(~???!u????@)?St$????1K?v??:Preprocessing2F
Iterator::Modelh??s???!0??@)?~j?t?h?1?yNȳ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.