?   *	???????@2?
VIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2 $???~???!?)?^;LC@)$???~???1?)?^;LC@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat "??u????!k*??20@)EGr????1???0Y-@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat ??ܵ???!?[?eb.@)-!?lV??1??.)o?*@:Preprocessing2?
[Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip ???ZӼ??!?r?I??9@)0*??D??1?1????@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle ??(\????!zV?@5?E@)??JY?8??1?f??o@:Preprocessing2l
5Iterator::Model::MaxIntraOpParallelism::Prefetch::Map4??@????!??6?`?I@)Dio??ɴ?1^???E@:Preprocessing2?
rIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip ?\?C????!????#6@)]?Fx??1???0zo@:Preprocessing2?
mIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2 =?U?????!??<*q?@)=?U?????1??<*q?@:Preprocessing2u
>Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2???QI??!9׹??AG@)??	h"l??1??*v@:Preprocessing2?
yIterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[0]::ForeverRepeat::FromTensor@??j+????!?`ca?7??)??j+????1?`ca?7??:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor@??ݓ????!n8??jd??)??ݓ????1n8??jd??:Preprocessing2?
|Iterator::Model::MaxIntraOpParallelism::Prefetch::Map::BatchV2::Shuffle::ParallelMapV2::Zip[1]::ParallelMapV2::Zip[0]::Range ??ZӼ???!,Dd?a6??)??ZӼ???1,Dd?a6??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch9??v????!<?fՆ|??)9??v????1<?fՆ|??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism46<???!E[C<??)?g??s?u?1?
@?B???:Preprocessing2F
Iterator::ModelM?O???!?Rm?)??)ŏ1w-!_?1?t??Z???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q݌@{?a"@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.JLAPTOP-4THGTL0B: Failed to load libcupti (is it installed and accessible?)