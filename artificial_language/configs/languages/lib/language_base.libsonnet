local sentence_length_sampler = import "empirical_distribution_sampler.libsonnet";

{
    vocab_size: 16000,
    sentence_length_sampler: sentence_length_sampler,
}
