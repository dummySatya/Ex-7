#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <sndfile.h>
#include <vector>
#include <samplerate.h>

std::vector<float> readAudioFile(const char *filename)
{
    SF_INFO sfinfo;
    SNDFILE *infile = sf_open(filename, SFM_READ, &sfinfo);
    if (!infile)
    {
        std::cerr << "Error opening file: " << sf_strerror(nullptr) << std::endl;
        return {};
    }

    // Read the audio data into a vector
    std::vector<float> samples(sfinfo.frames * sfinfo.channels);
    sf_readf_float(infile, samples.data(), sfinfo.frames);
    sf_close(infile);

    // Convert to mono if stereo (average the channels)
    std::vector<float> monoSamples;
    if (sfinfo.channels > 1)
    {
        monoSamples.resize(sfinfo.frames);
        for (int i = 0; i < sfinfo.frames; ++i)
        {
            monoSamples[i] = (samples[i * 2] + samples[i * 2 + 1]) / 2.0f;
        }
    }
    else
    {
        monoSamples = samples; // Already mono
    }

    return monoSamples;
}

int main()
{
    // check providers
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");

    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Create session
    Ort::Session session(env, "./rfft_model.onnx", session_options);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Input tensor
    std::vector<float> input_tensor_values = readAudioFile("../audio/small2.wav");

    int inp_shape = input_tensor_values.size();

    std::vector<int64_t> input_shape = {inp_shape};

    // Create input tensor with allocated buffer
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    // Output tensor
    int out_shape = inp_shape % 2 == 0 ? inp_shape / 2 + 1 : (inp_shape + 1) / 2;

    std::vector<float> output_tensor_values(out_shape);

    std::vector<int64_t> output_shape = {out_shape};

    // Create output tensor with allocated buffer
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        output_tensor_values.data(),
        output_tensor_values.size(),
        output_shape.data(),
        output_shape.size());

    // Input and output names (found from Netron)
    const char *input_names[] = {"l_audio_"};
    const char *output_names[] = {"abs_1"};

    // Run the model
    session.Run(Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, &output_tensor, 1);

    // Retrieve output data
    float *output_arr = output_tensor.GetTensorMutableData<float>();

    // Printing just first 10 elements for comparison
    std::cout << "Output values: ";
    for (int i = 0; i < 10; i++)
    {
        std::cout << output_arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}