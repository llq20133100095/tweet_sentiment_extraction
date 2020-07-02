"""
papre: MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification
"""
import logging

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.layers as L

from transformers import RobertaConfig, TFBertMainLayer, TFRobertaPreTrainedModel
from transformers.modeling_tf_roberta import TFRobertaEmbeddings
from transformers.tokenization_utils import BatchEncoding
from transformers.modeling_tf_utils import shape_list
from transformers.modeling_tf_bert import TFBertLayer, TFBertPooler
# from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
# from .modeling_tf_bert import TFBertEmbeddings, TFBertMainLayer, gelu
# from .modeling_tf_utils import TFPreTrainedModel, get_initializer, shape_list


logger = logging.getLogger(__name__)

class TFBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = [TFBertLayer(config, name="layer_._{}".format(i)) for i in range(config.num_hidden_layers)]

    def call(self, inputs, inputs2=None, l=None, mix_layer=1000, training=False):
        hidden_states, attention_mask, head_mask = inputs
        if inputs2 is not None:
            hidden_states2, attention_mask2, head_mask2 = inputs2

        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module([hidden_states, attention_mask, head_mask[i]], training=training)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if inputs2 is not None:
                    layer_outputs2 = layer_module([hidden_states2, attention_mask2, head_mask2[i]], training=training)
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if inputs2 is not None:
                    hidden_states = l * hidden_states + (1-l) * hidden_states2

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module([hidden_states, attention_mask, head_mask[i]], training=training)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class TFRobertaMainLayer(TFBertMainLayer):
    """
    Same as TFBertMainLayer but uses TFRobertaEmbeddings.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.num_hidden_layers = config.num_hidden_layers

        self.encoder = TFBertEncoder(config, name="encoder")
        self.pooler = TFBertPooler(config, name="pooler")
        self.embeddings = TFRobertaEmbeddings(config, name="embeddings")

    def get_input_embeddings(self):
        return self.embeddings

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def call(
            self,
            inputs,
            inputs2=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            attention_mask2=None,
            token_type_ids2=None,
            position_ids2=None,
            head_mask2=None,
            inputs_embeds2=None,
            training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            assert len(inputs) <= 6, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            assert len(inputs) <= 6, "Too many inputs."
        else:
            input_ids = inputs

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers

        if inputs2 is not None:
            if isinstance(inputs2, (tuple, list)):
                input_ids2 = inputs2[0]
                attention_mask2 = inputs2[1] if len(inputs2) > 1 else attention_mask2
                token_type_ids2 = inputs2[2] if len(inputs2) > 2 else token_type_ids2
                position_ids2 = inputs2[3] if len(inputs2) > 3 else position_ids2
                head_mask2 = inputs2[4] if len(inputs2) > 4 else head_mask2
                inputs_embeds2 = inputs2[5] if len(inputs2) > 5 else inputs_embeds2
                assert len(inputs) <= 6, "Too many inputs."
            elif isinstance(inputs2, (dict, BatchEncoding)):
                input_ids2 = inputs2.get("input_ids")
                attention_mask2 = inputs.get("attention_mask", attention_mask2)
                token_type_ids2 = inputs.get("token_type_ids", token_type_ids2)
                position_ids2 = inputs.get("position_ids", position_ids2)
                head_mask2 = inputs.get("head_mask", head_mask2)
                inputs_embeds2 = inputs.get("inputs_embeds", inputs_embeds2)
                assert len(inputs) <= 6, "Too many inputs."
            else:
                input_ids2 = inputs2

            if input_ids2 is not None and inputs_embeds2 is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids2 is not None:
                input_shape2 = shape_list(input_ids2)
            elif inputs_embeds is not None:
                input_shape2 = shape_list(inputs_embeds2)[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            if attention_mask2 is None:
                attention_mask2 = tf.fill(input_shape2, 1)
            if token_type_ids is None:
                token_type_ids2 = tf.fill(input_shape2, 0)

            extended_attention_mask2 = attention_mask2[:, tf.newaxis, tf.newaxis, :]

            extended_attention_mask2 = tf.cast(extended_attention_mask2, tf.float32)
            extended_attention_mask2 = (1.0 - extended_attention_mask2) * -10000.0

            # if head_mask2 is not None:
            #     raise NotImplementedError
            # else:
            #     head_mask2 = [None] * self.num_hidden_layers


            embedding_output = self.embeddings([input_ids, position_ids, token_type_ids, inputs_embeds], training=training)
            embedding_output2 = self.embeddings([input_ids2, position_ids2, token_type_ids2, inputs_embeds2], training=training)

            return embedding_output, extended_attention_mask, head_mask, embedding_output2, extended_attention_mask2, head_mask

        else:
            embedding_output = self.embeddings([input_ids, position_ids, token_type_ids, inputs_embeds], training=training)
            return embedding_output, extended_attention_mask, head_mask

    def call_run(self, embedding_output, extended_attention_mask, head_mask,
                 embedding_output2=None, extended_attention_mask2=None, head_mask2=None,
                 l=None, mix_layer=1000, training=False):
        if embedding_output2 is not None:
            encoder_outputs = self.encoder([embedding_output, extended_attention_mask, head_mask],
                                           [embedding_output2, extended_attention_mask2, head_mask2],
                                           l=l, mix_layer=mix_layer, training=training)
        else:
            encoder_outputs = self.encoder([embedding_output, extended_attention_mask, head_mask], training=training)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        return outputs


class RoBertQAModel(TFRobertaPreTrainedModel):
    DROPOUT_RATE = 0.1
    NUM_HIDDEN_STATES = 2

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.robert = TFRobertaMainLayer(config, name="roberta")

        self.dropout = L.Dropout(self.DROPOUT_RATE)
        self.conv1d_128 = L.Conv1D(128, 2, padding='same')
        self.conv1d_64 = L.Conv1D(64, 2, padding='same')
        self.leakyreLU = L.LeakyReLU()
        self.dense = L.Dense(1, dtype='float32')
        self.flatten = L.Flatten()

        self.dropout_2 = L.Dropout(self.DROPOUT_RATE)
        self.conv1d_128_2 = L.Conv1D(128, 2, padding='same')
        self.conv1d_64_2 = L.Conv1D(64, 2, padding='same')
        self.leakyreLU_2 = L.LeakyReLU()
        self.dense_2 = L.Dense(1, dtype='float32')
        self.flatten_2 = L.Flatten()

    @tf.function
    def call(self, inputs, inputs2=None, l=None, mix_layer=1000, **kwargs):
        if inputs2 is not None:
            embedding_output, extended_attention_mask, head_mask, embedding_output2, extended_attention_mask2, head_mask2 \
                = self.robert(inputs, inputs2, **kwargs)

            y_pred = self.call_run(embedding_output, extended_attention_mask, head_mask, embedding_output2, extended_attention_mask2, head_mask2,
                                   l=l, mix_layer=mix_layer, training=kwargs.get("training", False))
            return embedding_output, extended_attention_mask, head_mask, embedding_output2, extended_attention_mask2, head_mask2, y_pred

        else:
            embedding_output, extended_attention_mask, head_mask = self.robert(inputs, **kwargs)
            y_pred = self.call_run(embedding_output, extended_attention_mask, head_mask, training=kwargs.get("training", False))
            return embedding_output, extended_attention_mask, head_mask, y_pred

    @tf.function
    def call_run(self, embedding_output, extended_attention_mask, head_mask,
                 embedding_output2=None, extended_attention_mask2=None, head_mask2=None,
                 l=None, mix_layer=1000, training=False):

        hidden_states = []
        if embedding_output2 is not None:
            _, _, hidden_states = self.robert.call_run(embedding_output, extended_attention_mask, head_mask,
                                                       embedding_output2, extended_attention_mask2, head_mask2,
                                                       l=l, mix_layer=mix_layer, training=training)
        else:
            _, _, hidden_states = self.robert.call_run(embedding_output, extended_attention_mask, head_mask, training=training)

        x1 = self.dropout(hidden_states[-1], training=training)
        x1 = self.conv1d_128(x1)
        x1 = self.leakyreLU(x1)
        x1 = self.conv1d_64(x1)
        x1 = self.dense(x1)
        start_logits = self.flatten(x1)
        start_logits = L.Activation('softmax')(start_logits)

        x2 = self.dropout_2(hidden_states[-2], training=training)
        x2 = self.conv1d_128_2(x2)
        x2 = self.leakyreLU_2(x2)
        x2 = self.conv1d_64_2(x2)
        x2 = self.dense_2(x2)
        end_logits = self.flatten_2(x2)
        end_logits = L.Activation('softmax')(end_logits)
        return start_logits, end_logits

    @tf.function
    def adversarial(self, x1, extended_attention_mask, head_mask, y_true, loss_fn):
        """
        Adversarial training
        """
        y_pred = self.call_run(x1, extended_attention_mask, head_mask, training=True)
        loss = (loss_fn(y_true[0], y_pred[0]) + loss_fn(y_true[1], y_pred[1])) / 2
        perturb = tf.gradients(loss, x1, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)[0]

        # reciprocal in l2 normal
        perturb = 0.02 * tf.math.l2_normalize(tf.stop_gradient(perturb), axis=1)
        x1 = x1 + perturb

        # adv_loss
        y_pred = self.call_run(x1, extended_attention_mask, head_mask, training=True)
        adv_loss = loss_fn(y_true[0], y_pred[0]) + loss_fn(y_true[1], y_pred[1])
        return adv_loss

    @tf.function
    def virtual_adversarial(self, x1, extended_attention_mask, head_mask, y_pred, power_iterations=1, p_mult=0.02):
        bernoulli = tfp.distributions.Bernoulli
        prob1 = tf.clip_by_value((y_pred[0] + y_pred[1]) / 2, 1e-7, 1. - 1e-7)
        prob_dist1 = bernoulli(probs=prob1)

        # generate virtual adversarial perturbation
        d1 = tf.keras.backend.random_uniform(shape=tf.shape(x1), dtype=tf.dtypes.float32)
        for _ in range(power_iterations):
            d1 = (0.02) * tf.math.l2_normalize(d1, axis=1)
            y_pred1 = self.call_run(x1 + d1, extended_attention_mask, head_mask, training=True)
            p_prob1 = tf.clip_by_value((y_pred1[0] + y_pred1[1]) / 2, 1e-7, 1. - 1e-7)
            kl1 = tfp.distributions.kl_divergence(prob_dist1, bernoulli(probs=p_prob1), allow_nan_stats=False)

            gradient1 = tf.gradients(kl1, [d1], aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)[0]
            d1 = tf.stop_gradient(gradient1)
        d1 = p_mult * tf.math.l2_normalize(d1, axis=1)
        tf.stop_gradient(prob1)

        # virtual adversarial loss
        y_pred1 = self.call_run(x1 + d1, extended_attention_mask, head_mask, training=True)
        p_prob1 = tf.clip_by_value((y_pred1[0] + y_pred1[1]) / 2, 1e-7, 1. - 1e-7)
        v_adv_loss1 = tfp.distributions.kl_divergence(prob_dist1, bernoulli(probs=p_prob1), allow_nan_stats=False)
        return tf.reduce_mean(v_adv_loss1)