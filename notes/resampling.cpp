#include <iostream>
#include <sndfile.h>
#include <vector>
#include <samplerate.h>

std::vector<float> readAudioFile(const char *filename, int targetSampleRate = 22050)
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
    // Resample audio to target sample rate (22kHz)
    int originalSampleRate = sfinfo.samplerate;
    if (originalSampleRate == targetSampleRate)
    {
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
    int error = src_simple(&srcData, SRC_SINC_BEST_QUALITY, 1);
    if (error != 0)
    {
        std::cerr << "Error in resampling: " << src_strerror(error) << std::endl;
        return {};
    }

    return resampledAudio;
}
