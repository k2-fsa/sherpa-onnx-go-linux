package sherpa_onnx

// #include <stdlib.h>
// #include "c-api.h"
import "C"
import "unsafe"

// OnlinePunctuationModelConfig configures the CNN-BiLSTM punctuation model.
type OnlinePunctuationModelConfig struct {
	CnnBilstm  string
	BpeVocab   string
	NumThreads C.int
	Debug      C.int
	Provider   string
}

// OnlinePunctuationConfig wraps the model config.
type OnlinePunctuationConfig struct {
	Model OnlinePunctuationModelConfig
}

// OnlinePunctuation holds a pointer to the C implementation.
type OnlinePunctuation struct {
	impl *C.struct_SherpaOnnxOnlinePunctuation
}

// NewOnlinePunctuation creates an online punctuation processor.
func NewOnlinePunctuation(config *OnlinePunctuationConfig) *OnlinePunctuation {
	cfg := C.struct_SherpaOnnxOnlinePunctuationConfig{}
	cfg.model.cnn_bilstm = C.CString(config.Model.CnnBilstm)
	defer C.free(unsafe.Pointer(cfg.model.cnn_bilstm))

	cfg.model.bpe_vocab = C.CString(config.Model.BpeVocab)
	defer C.free(unsafe.Pointer(cfg.model.bpe_vocab))

	cfg.model.num_threads = config.Model.NumThreads
	cfg.model.debug = config.Model.Debug
	cfg.model.provider = C.CString(config.Model.Provider)
	defer C.free(unsafe.Pointer(cfg.model.provider))

	impl := C.SherpaOnnxCreateOnlinePunctuation(&cfg)
	if impl == nil {
		return nil
	}
	punc := &OnlinePunctuation{}
	punc.impl = impl
	return punc
}

// DeleteOnlinePunctuation frees the C resources.
func DeleteOnlinePunctuation(punc *OnlinePunctuation) {
	C.SherpaOnnxDestroyOnlinePunctuation(punc.impl)
	punc.impl = nil
}

// AddPunct adds punctuation and truecasing to the input text.
func (punc *OnlinePunctuation) AddPunct(text string) string {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	p := C.SherpaOnnxOnlinePunctuationAddPunct(punc.impl, cText)
	defer C.SherpaOnnxOnlinePunctuationFreeText(p)

	return C.GoString(p)
}
