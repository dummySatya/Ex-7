#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <sndfile.h>
#include <vector>
#include <samplerate.h>

std::vector<float> readAudioFile(const char *filename, int targetSampleRate = 22050) {
    SF_INFO sfinfo;
    SNDFILE *infile = sf_open(filename, SFM_READ, &sfinfo);
    if (!infile) {
        std::cerr << "Error opening file: " << sf_strerror(nullptr) << std::endl;
        return {};
    }

    // Read the audio data into a vector
    std::vector<float> samples(sfinfo.frames * sfinfo.channels);
    sf_readf_float(infile, samples.data(), sfinfo.frames);
    sf_close(infile);

    // Convert to mono if stereo (average the channels)
    std::vector<float> monoSamples;
    if (sfinfo.channels > 1) {
        monoSamples.resize(sfinfo.frames);
        for (int i = 0; i < sfinfo.frames; ++i) {
            // Average the stereo channels (simple mono conversion)
            monoSamples[i] = (samples[i * 2] + samples[i * 2 + 1]) / 2.0f;
        }
    } else {
        monoSamples = samples; // Already mono, so no need to change
    }

    return monoSamples;

    // Resample audio to target sample rate (22kHz)
    int originalSampleRate = sfinfo.samplerate;
    if (originalSampleRate == targetSampleRate) {
        // If original and target sample rates are the same, no need to resample
        return monoSamples;
    }

    // Resample the audio
    int outputLength = static_cast<int>(monoSamples.size() * static_cast<float>(targetSampleRate) / originalSampleRate);
    std::vector<float> resampledAudio(outputLength);

    // Set up the SRC_DATA structure for libsamplerate
    SRC_DATA srcData;
    srcData.data_in = monoSamples.data();
    srcData.input_frames = monoSamples.size();
    srcData.data_out = resampledAudio.data();
    srcData.output_frames = outputLength;
    srcData.src_ratio = static_cast<float>(targetSampleRate) / originalSampleRate;

    // Perform resampling
    int error = src_simple(&srcData, SRC_SINC_BEST_QUALITY,1);
    if (error != 0) {
        std::cerr << "Error in resampling: " << src_strerror(error) << std::endl;
        return {};
    }

    return resampledAudio; // Return the resampled mono audio data
}


int main()
{
    // check providers
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");

    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(10);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Create session
    Ort::Session session(env, "./rfft_model.onnx", session_options);
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<float> input_tensor_values = readAudioFile("../audio/small2.wav"); // Example input, replace with actual audio data
    int inp_shape = input_tensor_values.size();
    std::cout<<inp_shape<<std::endl;
    // inp_shape = 661127;
    std::vector<int64_t> input_shape = {inp_shape};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memoryInfo, 
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    // Prepare output tensor
    int out_shape = inp_shape % 2 == 0? inp_shape/2 + 1 : (inp_shape + 1) / 2;
    // out_shape = 330564;
    std::vector<float> output_tensor_values(out_shape); // Allocate space for 14405 elements
    std::vector<int64_t> output_shape = {out_shape};

    // Create output tensor with allocated buffer
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        output_tensor_values.data(),
        output_tensor_values.size(),
        output_shape.data(),
        output_shape.size());    // Run the model
    const char *input_names[] = {"l_audio_"}; // Match with input name from your ONNX model
    const char *output_names[] = {"abs_1"};   // Match with output name from your ONNX model

    session.Run(Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, &output_tensor, 1);

    // Retrieve output data
    float *output_arr = output_tensor.GetTensorMutableData<float>();

    // Process output data (for example, print first 10 elements)
    std::cout << "Output values: ";
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << output_arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}