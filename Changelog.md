# Changelog

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
