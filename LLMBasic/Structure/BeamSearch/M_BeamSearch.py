import tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PhrasalConstraint,
    )

def m_beam_search():
    tokenizer = AutoTokenizer.from_pretrained('/code/model/t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained('/code/model/t5-base')

    input = 'translate English to German: How old are you?'

    input_ids = tokenizer.encode(input,return_tensors='pt')

    """[constraints]对需要出现的token做tokenize"""
    # force_words = ['Sie']
    # # force_words_ids = tokenizer(force_words,add_special_tokens=False).input_ids
    # outputs = model.generate(
    #     input_ids,
    #     force_words_ids=force_words_ids,
    #     num_beams=10,
    #     num_return_sequences=1,
    #     no_repeat_ngram_size=1,
    #     remove_invalid_values=True,
    #     max_length = 20,
    # )

    constraints = [
        PhrasalConstraint(
            token_ids=tokenizer('Sie',add_special_tokens=False).input_ids
        )
    ]
    outputs = model.generate(
        input_ids,
        constraints=constraints,
        num_beams=10,
        num_return_sequences=1,
        no_repeat_ngram_size=1,
        remove_invalid_values=True,
        max_length = 20,
    )

    print("Output:\n" + 100 *'-')
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    m_beam_search()