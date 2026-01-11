#include "Utils.h"

// ========================================
// Shape & Tensor Utility Function(s)
// ========================================
bool Utils::IsVolumeOverflow(const std::vector<int>& nums)
{
	long long volume = 1;
	for (const auto& num : nums)
	{
		volume = volume * static_cast<long long>(num);
		if (volume > INT_MAX)
		{
			return true;
		}
	}
	return false;
}

int Utils::ShapeToVolume(const std::vector<int>& nums)
{
	if (Utils::IsVolumeOverflow(nums))
	{
		throw std::overflow_error("[Tensor-Utils] Volume Computation failed: shape too large, potential overflow.");
	}

	int volume = 1;
	for (const auto& num : nums)
	{
		volume *= num;
	}
	return volume;
}

std::vector<int> Utils::ShapeToStrides(const std::vector<int>& shape)
{
	if (Utils::IsVolumeOverflow(shape))
	{
		throw std::overflow_error("[Tensor-Utils] Stride Computation failed: shape too large, potential overflow.");
	}

	int nd = static_cast<int>(shape.size());
	
	std::vector<int> strides(nd);
	int volume = 1;

	for(int i = (nd - 1); i >= 0; i--)
	{
		strides[i] = volume;
		volume *= shape[i];
	}
	return strides;
}

int Utils::FlatIndex(const std::vector<int>& shape, const std::vector<int>& tensor_index)
{
	if(tensor_index.size() != shape.size())
	{
		throw std::invalid_argument("[Tensor-Utils] Flat Index Computation failed: array size mismatch between index and shape.");
	}
	
	if(!Utils::IsBounded(tensor_index, shape, -1, true))
	{
		throw std::out_of_range("[Tensor-Utils] Flat Index Computation failed: index values out of bound from shape.");
	}
	
	std::vector<int> strides = Utils::ShapeToStrides(shape);
	int flat_index = 0;

	for(size_t i = 0; i < shape.size(); i++)
	{
		flat_index += (tensor_index[i] * strides[i]);
	}
	return flat_index;
}

std::vector<int> Utils::TensorIndex(const std::vector<int>& shape, int flat_index)
{
	if (flat_index >= Utils::ShapeToVolume(shape))
	{
		throw std::invalid_argument("[Tensor-Utils] Tensor Index Computation failed: out of bound flat index.");
	}

	int nd = static_cast<int>(shape.size());
	
	std::vector<int> strides = Utils::ShapeToStrides(shape);
	std::vector<int> tensor_index(nd);
	
	for(size_t i = 0; i < nd; i++)
	{
		tensor_index[i] = flat_index / strides[i];
		flat_index %= strides[i];
	}
	return tensor_index;
}


// ========================================
// Broadcasting & Convolution Utility Function(s)
// ========================================
bool Utils::IsBroadcastCompatible(const std::vector<int>& shape_1, const std::vector<int>& shape_2)
{
	int dimension_1 = static_cast<int>(shape_1.size());
	int dimension_2 = static_cast<int>(shape_2.size());

	while (dimension_1 && dimension_2)
	{
		dimension_1--;
		dimension_2--;

		if (shape_1[dimension_1] != 1 && shape_2[dimension_2] != 1 && shape_1[dimension_1] != shape_2[dimension_2])
		{
			return false;
		}
	}
	return true;
}

bool Utils::IsConvolveCompatible(const std::vector<int>& main_shape, const std::vector<int>& filter_shape)
{
	int main_dimension = static_cast<int>(main_shape.size());
	int filter_dimension = static_cast<int>(filter_shape.size());

	if (main_dimension < filter_dimension)
	{
		return false;
	}

	while (main_dimension && filter_dimension)
	{
		main_dimension--;
		filter_dimension--;

		if (main_shape[main_dimension] < filter_shape[filter_dimension])
		{
			return false;
		}
	}
	return true;
}

std::vector<int> Utils::BroadcastShape(const std::vector<int>& shape_1, const std::vector<int>& shape_2)
{
	if (!Utils::IsBroadcastCompatible(shape_1, shape_2))
	{
		throw std::invalid_argument("[Tensor-Utils] Broadcast-Shape Computation failed: shapes are not compatible for broadcasting.");
	}

	int dimension_1 = static_cast<int>(shape_1.size());
	int dimension_2 = static_cast<int>(shape_2.size());

	std::vector<int> broadcast_shape;

	while (dimension_1 && dimension_2)
	{
		dimension_1--;
		dimension_2--;

		if (shape_1[dimension_1] == shape_2[dimension_2])
		{
			broadcast_shape.push_back(shape_1[dimension_1]);
		}
		else if (shape_1[dimension_1] == 1)
		{
			broadcast_shape.push_back(shape_2[dimension_2]);
		}
		else
		{
			broadcast_shape.push_back(shape_1[dimension_1]);
		}
	}
	while (--dimension_1 >= 0)
	{
		broadcast_shape.push_back(shape_1[dimension_1]);
	}
	while (--dimension_2 >= 0)
	{
		broadcast_shape.push_back(shape_2[dimension_2]);
	}

	std::reverse(broadcast_shape.begin(), broadcast_shape.end());

	return broadcast_shape;
}

std::vector<int> Utils::ConvolvedFeatureShape(const std::vector<int>& main_shape, const std::vector<int>& filter_shape, const std::vector<int>& strides)
{
	if (!Utils::IsConvolveCompatible(main_shape, filter_shape))
	{
		throw std::invalid_argument("[Tensor-Utils] Convolved Feature-Shape Computation failed: shapes are not compatible for convolution.");
	}

	if (main_shape.size() != filter_shape.size())
	{
		throw std::invalid_argument("[Tensor-Utils] Convolved Feature-Shape Computation failed: main_shape and filter_shape must have same number of dimensions.");
	}

	if (main_shape.size() != strides.size())
	{
		throw std::invalid_argument("[Tensor-Utils] Convolved Feature-Shape Computation failed: main_shape and strides must have same number of dimensions.");
	}

	int nd = static_cast<int>(main_shape.size());

	std::vector<int> conv_shape(nd, 0);

	for (size_t i = 0; i < nd; i++)
	{
		conv_shape[i] = ((main_shape[i] - filter_shape[i]) / strides[i]) + 1;
	}

	return conv_shape;
}


// ========================================
// Set Operation Function(s)
// ========================================
std::vector<int> Utils::FindRangeComplement(const std::vector<int>& nums, std::pair<int, int> bounds)
{
	if (bounds.first >= bounds.second)
	{
		throw std::invalid_argument("[Tensor-Utils] Get Missing Value failed: inappropriate bounds for generating missing value(s).");
	}

	int range = bounds.second - bounds.first;

	std::vector<bool> is_present(range, false);
	for (auto& num : nums)
	{
		if (num < bounds.first || num >= bounds.second)
		{
			continue;
		}
		int index = num - bounds.first;
		is_present[index] = true;
	}

	std::vector<int> result;
	for (size_t i = 0; i < range; i++)
	{
		if (!is_present[i])
		{
			result.push_back(i + bounds.first);
		}
	}

	return result;
}
