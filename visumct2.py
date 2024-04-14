import ctranslate2
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("model2/source.spm")

source = sp.encode("Hello world!", out_type=str)

translator = ctranslate2.Translator("model2")
results = translator.translate_batch([source])

output = sp.decode(results[0].hypotheses[0])
print(output)
