local language_base = import "lib/language_base.libsonnet";

{
    type: "parenthesis",
    parenthesis_type: "nesting",
    language: {
        type: "random_walk",
        vocab_size: std.ceil(language_base["vocab_size"] / 2),
        sentence_length_sampler: language_base["sentence_length_sampler"],
    },
}
