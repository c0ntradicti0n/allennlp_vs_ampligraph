# Simple Tagger

### Simple Tagger with nym embeddings

```
```
refuses to work at some loss-level much above without knowledge-embeddings.
### Simple Tagger without nym embeddings
```
./train_difference.sh simple_tagger_without_nyms.json  
```
works fine. Learns epoch for epoch more and more
```
2019-10-15 12:45:23,697 - INFO - allennlp.training.trainer - Beginning training.
2019-10-15 12:45:23,697 - INFO - allennlp.training.trainer - Epoch 0/39
2019-10-15 12:45:23,697 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 1919.684
2019-10-15 12:45:23,739 - INFO - allennlp.training.trainer - Training
accuracy: 0.1656, accuracy3: 0.3559, loss: 3.5716 ||: 100%|##########| 125/125 [00:37<00:00,  3.32it/s]
2019-10-15 12:46:01,372 - INFO - allennlp.training.trainer - Validating
accuracy: 0.1852, accuracy3: 0.3841, loss: 3.2306 ||: 100%|##########| 31/31 [00:02<00:00, 14.47it/s]
2019-10-15 12:46:03,515 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:46:03,515 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  1919.684  |       N/A
2019-10-15 12:46:03,516 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.356  |     0.384
2019-10-15 12:46:03,516 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.166  |     0.185
2019-10-15 12:46:03,517 - INFO - allennlp.training.tensorboard_writer - loss          |     3.572  |     3.231
2019-10-15 12:46:03,540 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:46:03,546 - INFO - allennlp.training.trainer - Epoch duration: 0:00:39.848754
2019-10-15 12:46:03,548 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:25:54
2019-10-15 12:46:03,548 - INFO - allennlp.training.trainer - Epoch 1/39
2019-10-15 12:46:03,548 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2232.72
2019-10-15 12:46:03,602 - INFO - allennlp.training.trainer - Training
accuracy: 0.2511, accuracy3: 0.4494, loss: 3.0232 ||: 100%|##########| 125/125 [00:38<00:00,  3.23it/s]
2019-10-15 12:46:42,335 - INFO - allennlp.training.trainer - Validating
accuracy: 0.3684, accuracy3: 0.5976, loss: 2.4891 ||: 100%|##########| 31/31 [00:01<00:00, 20.92it/s]
2019-10-15 12:46:43,818 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:46:43,818 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2232.720  |       N/A
2019-10-15 12:46:43,819 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.449  |     0.598
2019-10-15 12:46:43,819 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.251  |     0.368
2019-10-15 12:46:43,820 - INFO - allennlp.training.tensorboard_writer - loss          |     3.023  |     2.489
2019-10-15 12:46:43,841 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:46:43,850 - INFO - allennlp.training.trainer - Epoch duration: 0:00:40.301615
2019-10-15 12:46:43,850 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:25:22
2019-10-15 12:46:43,850 - INFO - allennlp.training.trainer - Epoch 2/39
2019-10-15 12:46:43,850 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2324.584
2019-10-15 12:46:43,918 - INFO - allennlp.training.trainer - Training
accuracy: 0.4859, accuracy3: 0.6952, loss: 2.0343 ||: 100%|##########| 125/125 [00:43<00:00,  2.91it/s]
2019-10-15 12:47:26,928 - INFO - allennlp.training.trainer - Validating
accuracy: 0.5964, accuracy3: 0.7851, loss: 1.6124 ||: 100%|##########| 31/31 [00:01<00:00, 18.35it/s]
2019-10-15 12:47:28,618 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:47:28,618 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2324.584  |       N/A
2019-10-15 12:47:28,619 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.695  |     0.785
2019-10-15 12:47:28,619 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.486  |     0.596
2019-10-15 12:47:28,620 - INFO - allennlp.training.tensorboard_writer - loss          |     2.034  |     1.612
2019-10-15 12:47:28,637 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:47:28,646 - INFO - allennlp.training.trainer - Epoch duration: 0:00:44.796049
2019-10-15 12:47:28,646 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:25:41
2019-10-15 12:47:28,646 - INFO - allennlp.training.trainer - Epoch 3/39
2019-10-15 12:47:28,646 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2370.756
2019-10-15 12:47:28,712 - INFO - allennlp.training.trainer - Training
accuracy: 0.6571, accuracy3: 0.8196, loss: 1.3888 ||: 100%|##########| 125/125 [00:40<00:00,  3.08it/s]
2019-10-15 12:48:09,280 - INFO - allennlp.training.trainer - Validating
accuracy: 0.7213, accuracy3: 0.8532, loss: 1.1983 ||: 100%|##########| 31/31 [00:01<00:00, 19.22it/s]
2019-10-15 12:48:10,894 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:48:10,895 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2370.756  |       N/A
2019-10-15 12:48:10,895 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.820  |     0.853
2019-10-15 12:48:10,895 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.657  |     0.721
2019-10-15 12:48:10,896 - INFO - allennlp.training.tensorboard_writer - loss          |     1.389  |     1.198
2019-10-15 12:48:10,917 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:48:10,925 - INFO - allennlp.training.trainer - Epoch duration: 0:00:42.278537
2019-10-15 12:48:10,925 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:25:05
2019-10-15 12:48:10,925 - INFO - allennlp.training.trainer - Epoch 4/39
2019-10-15 12:48:10,925 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2420.084
2019-10-15 12:48:11,003 - INFO - allennlp.training.trainer - Training
accuracy: 0.7611, accuracy3: 0.8917, loss: 1.0077 ||: 100%|##########| 125/125 [00:42<00:00,  2.96it/s]
2019-10-15 12:48:53,291 - INFO - allennlp.training.trainer - Validating
accuracy: 0.7871, accuracy3: 0.8971, loss: 0.9419 ||: 100%|##########| 31/31 [00:01<00:00, 17.89it/s]
2019-10-15 12:48:55,025 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:48:55,025 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2420.084  |       N/A
2019-10-15 12:48:55,025 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.892  |     0.897
2019-10-15 12:48:55,026 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.761  |     0.787
2019-10-15 12:48:55,026 - INFO - allennlp.training.tensorboard_writer - loss          |     1.008  |     0.942
2019-10-15 12:48:55,043 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.
2019-10-15 12:48:55,050 - INFO - allennlp.training.trainer - Epoch duration: 0:00:44.124698
2019-10-15 12:48:55,050 - INFO - allennlp.training.trainer - Estimated training time remaining: 0:24:39
2019-10-15 12:48:55,050 - INFO - allennlp.training.trainer - Epoch 5/39
2019-10-15 12:48:55,050 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 2470.244
2019-10-15 12:48:55,117 - INFO - allennlp.training.trainer - Training
accuracy: 0.8307, accuracy3: 0.9313, loss: 0.7537 ||: 100%|##########| 125/125 [00:39<00:00,  3.17it/s]
2019-10-15 12:49:34,615 - INFO - allennlp.training.trainer - Validating
accuracy: 0.8269, accuracy3: 0.9214, loss: 0.7693 ||: 100%|##########| 31/31 [00:01<00:00, 20.66it/s]
2019-10-15 12:49:36,117 - INFO - allennlp.training.tensorboard_writer -                   Training |  Validation
2019-10-15 12:49:36,118 - INFO - allennlp.training.tensorboard_writer - cpu_memory_MB |  2470.244  |       N/A
2019-10-15 12:49:36,118 - INFO - allennlp.training.tensorboard_writer - accuracy3     |     0.931  |     0.921
2019-10-15 12:49:36,119 - INFO - allennlp.training.tensorboard_writer - accuracy      |     0.831  |     0.827
2019-10-15 12:49:36,119 - INFO - allennlp.training.tensorboard_writer - loss          |     0.754  |     0.769
2019-10-15 12:49:36,139 - INFO - allennlp.training.checkpointer - Best validation performance so far. Copying weights to './output/simple_tagger_without_nyms.json//best.th'.

```
# Span tagging with Conditional Random Field
### With nym embeddings
```./train_difference.sh simple_span_tagger_without.json```

gives error:

```
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
    RuntimeError: Invalid index in gather at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:657
     0%|          | 0/439 [00:01<?, ?it/s]

```

### Whithout nym embeddings
``` ./train_difference.sh span_tagger_with_nyms.json ```

works fine...

```
2019-10-15 12:40:55,864 - INFO - allennlp.training.trainer - Beginning training.
2019-10-15 12:40:55,865 - INFO - allennlp.training.trainer - Epoch 0/39
2019-10-15 12:40:55,865 - INFO - allennlp.training.trainer - Peak CPU memory usage MB: 1967.748
2019-10-15 12:40:55,905 - INFO - allennlp.training.trainer - Training
accuracy: 0.8366, accuracy3: 0.8453, precision-overall: 0.2904, recall-overall: 0.1447, f1-measure-overall: 0.1931, loss: 330.0508 ||:  83%|########3 | 365/439 [01:25<00:16,  4.42it/s]^
```