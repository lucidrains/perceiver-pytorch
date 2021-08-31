from perceiver_pytorch.perceiver_pytorch import PerceiverIO, fourier_encode
import torch
from typing import List, Iterable, Dict, Optional, Any
from einops import rearrange, repeat
from dataclasses import dataclass


@dataclass
class InputModality:
    name: str
    input_channels: int
    input_axis: int
    num_freq_bands: int
    max_freq: float
    freq_base: int = 2
    sin_only: bool = False
    fourier_encode: bool = True

    @property
    def input_dim(self) -> int:
        # Calculate the dimension of this modality.
        fourier_channels = self.input_axis * ((self.num_freq_bands * 2) + 1)
        fourier_channels = fourier_channels // 2 if self.sin_only else fourier_channels
        input_dim = fourier_channels + self.input_channels
        return input_dim


def modality_encoding(batch_size: int, axes, modality_index: int, num_modalities: int) -> torch.Tensor:
    """
    Return one-hot encoding of modality given num_modalities, batch size and axes.
    The result need to be compatible with the modality data for concatenation.
    :param modality_index:
    :param num_modalities:
    :return:
    """
    one_hot = torch.eye(num_modalities, num_modalities)[modality_index]
    to_expand = [batch_size]
    one_hot = one_hot.unsqueeze(0)
    for i, axis in enumerate(axes):
        one_hot = one_hot.unsqueeze(0)
        to_expand.append(axis)
    to_expand.append(num_modalities)

    one_hot = one_hot.expand(to_expand)
    return one_hot


class MultiPerceiver(torch.nn.Module):
    def __init__(
            self,
            modalities: Iterable[InputModality],
            fourier_encode_data: bool = True,
            input_channels: int = 3,
            output_channels: int = 12,
            forecast_steps: int = 48,
            sin_only: bool = False,
            output_shape: int = 32,
            **kwargs,
    ):
        """
        PerceiverIO made to work more specifically with timeseries images
        Not a recurrent model, so like MetNet somewhat, can optionally give a one-hot encoded vector for the future
        timestep
        Args:
            input_channels: Number of input channels
            forecast_steps: Number of forecast steps to make
            **kwargs:
        """
        super(MultiPerceiver, self).__init__()
        self.fourier_encode_data = fourier_encode_data
        self.forecast_steps = forecast_steps
        self.input_channels = input_channels
        self.sin_only = sin_only
        self.output_channels = output_channels
        self.modalities = {modality.name: modality for modality in modalities}
        # we encode modality with one hot encoding, so need one dim per modality:
        modality_encoding_dim = sum([1 for _ in modalities])
        # input_dim is the maximum dimension over all input modalities:
        input_dim = max(modality.input_dim for modality in modalities) + modality_encoding_dim
        # Pop dim
        self.max_modality_dim = input_dim
        kwargs.pop("dim")
        # Want toe logit_dim to be the same as the channels * width or height
        kwargs["logits_dim"] = output_shape * self.output_channels
        self.perceiver = PerceiverIO(dim=input_dim, **kwargs)

    def decode_output(self, data):
        pass

    def forward(self, multi_modality_data: Dict[str, torch.Tensor], mask=None, queries=None):
        batch_sizes = set()
        num_modalities = len(multi_modality_data)
        linearized_data = []

        for modality_index, modality_name in enumerate(sorted(multi_modality_data.keys())):
            assert (
                    modality_name in self.modalities
            ), f"modality {modality_name} was not defined in constructor"
            data = multi_modality_data[modality_name]
            modality = self.modalities[modality_name]
            b, *axis, _ = data.size()
            assert len(axis) == modality.input_axis, (
                f"input data must have the right number of  for modality {modality_name}. "
                f"Expected {modality.input_axis} while forward argument offered {len(axis)}"
            )
            batch_sizes.add(b)
            assert len(batch_sizes) == 1, "batch size must be the same across all modalities"
            enc_pos = []
            if self.fourier_encode_data:
                # calculate fourier encoded positions in the range of [-1, 1], for all axis

                axis_pos = list(
                    map(lambda size: torch.linspace(-1.0, 1.0, steps=size).type_as(data), axis)
                )
                pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
                enc_pos = fourier_encode(
                    pos,
                    modality.max_freq,
                    modality.num_freq_bands,
                    modality.freq_base,
                    sin_only=self.sin_only,
                )
                enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
                enc_pos = repeat(enc_pos, "... -> b ...", b=b)

            # Figure out padding for this modality, given max dimension across all modalities:
            padding_size = self.max_modality_dim - modality.input_dim - num_modalities

            padding = torch.zeros(size=data.size()[0:-1] + (padding_size,)).type_as(data)
            # concat to channels of data and flatten axis
            modality_encodings = modality_encoding(
                b, axis, modality_index, num_modalities
            ).type_as(data)
            to_concat = (
                (data, padding, enc_pos, modality_encodings)
                if len(enc_pos) > 0
                else (data, padding, modality_encodings)
            )
            data = torch.cat(to_concat, dim=-1)
            # concat to channels of data and flatten axis
            data = rearrange(data, "b ... d -> b (...) d")
            linearized_data.append(data)

        # Concatenate all the modalities:
        data = torch.cat(linearized_data, dim=1)

        # After this is the PerceiverIO backbone, still would need to decode it back to an image though
        # Should include the query shape here for the output we want, could be learned embeddings, repeated input frames of the same shape that is desired, etc.
        perceiver_output = self.perceiver.forward(data, mask, queries)

        # For multiple modalities, they are split after this beack into different tensors
        # For Sat images, we just want the images, not the other ones, so can leave it as is?

        # Have to decode back into future Sat image frames
        # Perceiver for 'pixel' postprocessing does nothing, or undoes the space2depth from before if just image
        # If doing depth2space, should split modalities again

        # Reshape to the correct output
        # This is how it is done in the official implementation, do a decoder query with cross attention, then just reshape the output
        # For timeseries, this is given as a query with T*H*W shape
        # For Flow Decoder, this is the same, except has a rescale factor
        perceiver_output = rearrange(
            perceiver_output, "b h (w c) -> b c h w", c=self.output_channels
        )

        return perceiver_output