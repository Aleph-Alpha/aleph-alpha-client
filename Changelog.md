# Changelog

## 2.1.0

### New feature

* Add new parameters to control how repetition penalties are applied for completion requests (see [docs](https://docs.aleph-alpha.com/api/#/paths/~1complete/post) for more information):
  * `penalty_bias`
  * `penalty_exceptions`
  * `penalty_exceptions_include_stop_sequences`

## 2.0.0

### Breaking change

* Make hosting parameter optional in semantic_embed on client. Changed order of parameters `hosting` and `request`.
  Should not be an issue if you're not using semantic_embed from the client directly or if you're using keyword args.

### Experimental feature

* Add experimental penalty parameters for completion

## 1.7.1

* Improved handling of text-based Documents in Q&A

## 1.7.0

* Introduce `semantic_embed` endpoint on client and model.
* Introduce timeout on client

## 1.6.0

* Introduce AlephAlphaModel as a more convenient alternative to direct usage of AlephAlphaClient

## 1.1.0

* Support for sending images to multimodal Models.

## 1.0.0

* Initial Release
