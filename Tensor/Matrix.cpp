#include "Matrix.h"

// ========================================
// [Private] Full Row/Column Check Method(s)
// ========================================
bool LinAlg::Matrix::IsFullColumnRank() const
{
	if (this->IsEmpty())
		return false;

	return this->Rank() == this->shape.second;
}

bool LinAlg::Matrix::IsFullRowRank() const
{
	if (this->IsEmpty())
		return false;

	return this->Rank() == this->shape.first;
}

// ========================================
// Matrix Constructor(s)
// ========================================
LinAlg::Matrix::Matrix(const std::vector<std::vector<double>>& _matrix)
{
	size_t rows = _matrix.size();
	size_t columns = 0;
	for (const auto& row : _matrix)
	{
		columns = std::max(columns, row.size());
	}

	this->shape = { static_cast<int>(rows), static_cast<int>(columns) };
	this->volume = (rows * columns);

	this->data.assign(rows, std::vector<double>(columns, 0.0));

	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < columns; j++)
		{
			this->data[i][j] = (j < _matrix[i].size()) ? _matrix[i][j] : 0;
		}
	}
}

LinAlg::Matrix::Matrix(const std::pair<int, int>& _shape, const double& _value)
{
	if (_shape.first <= 0 || _shape.second <= 0)
	{
		throw std::invalid_argument("[Matrix] Constructor failed: no. of row and column of a matrix must be > 0.");
	}

	if (!Utils::IsValidData<double>({ _value }))
	{
		throw std::invalid_argument("[Matrix] Constructor failed: invalid value.");
	}

	this->shape = { _shape.first, _shape.second };
	this->volume = (this->shape.first * this->shape.second);

	this->data.assign(this->shape.first, std::vector<double>(this->shape.second, _value));
}

LinAlg::Matrix::Matrix(const std::pair<int, int>& _shape, std::vector<double>& _data)
{
	if (_shape.first <= 0 || _shape.second <= 0)
	{
		throw std::invalid_argument("[Matrix] Constructor failed: no. of row and column of a matrix must be > 0.");
	}

	this->shape = { _shape.first, _shape.second };
	this->volume = (this->shape.first * this->shape.second);

	if (_data.size() != this->volume)
	{
		throw std::runtime_error("[Matrix] Constructor failed: volume mismatch between data-array and shape.");
	}

	this->data.assign(this->shape.first, std::vector<double>(this->shape.second, 0.0));

	for (int i = 0, k = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++, k++)
		{
			this->data[i][j] = _data[k];
		}
	}
}

// ========================================
// Special Matrix Initialization Method(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::Identity(const int& _n, const double& _scale)
{
    if (_n <= 0)
    {
        throw std::invalid_argument("[Matrix] Identity Matrix Build failed: row/column size of matrix must be > 0.");
    }

	std::vector<std::vector<double>> I_matrix(_n, std::vector<double>(_n, 0.0));

    for (int i = 0; i < _n; i++)
    {
		I_matrix[i][i] = _scale;
    }

    return LinAlg::Matrix(I_matrix);
}

LinAlg::Matrix LinAlg::Matrix::RandomUniform(const int& _rows, const int& _columns, const double& _min_value, const double& _max_value, std::optional<unsigned int> seed)
{
    if (_rows <= 0 || _columns <= 0)
    {
        throw std::invalid_argument("[Matrix] Random-Uniform Matrix Build failed: no. of rows or columns must be > 0.");
    }

	if (_min_value >= _max_value)
	{
		throw std::invalid_argument("[Matrix] Random-Uniform Matrix Build failed: minimum bound must be < maximum bound.");
	}

    std::mt19937 generator;
    if (seed.has_value())
    {
        generator.seed(seed.value());
    }
    else
    {
        std::random_device rd;
        generator.seed(rd());
    }

    std::uniform_real_distribution<double> dist(_min_value, _max_value);

	std::vector<std::vector<double>> uniform_matrix(_rows, std::vector<double>(_columns));

    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _columns; j++)
        {
			uniform_matrix[i][j] = dist(generator);
        }
    }

    return LinAlg::Matrix(uniform_matrix);
}

LinAlg::Matrix LinAlg::Matrix::RandomNormal(const int& _rows, const int& _columns, const double& _mean, const double& _std_dev, std::optional<unsigned int> seed)
{
	if (_rows <= 0 || _columns <= 0)
	{
		throw std::invalid_argument("[Matrix] Random-Normal Matrix Build failed: no. of rows or columns must be > 0.");
	}

	if (_std_dev < 0)
	{
		throw std::invalid_argument("[Matrix] Random-Normal Matrix Build failed: negative standard deviation found.");
	}

	std::mt19937 generator;
	if (seed.has_value())
	{
		generator.seed(seed.value());
	}
	else
	{
		std::random_device rd;
		generator.seed(rd());
	}

	std::normal_distribution<double> dist(_mean, _std_dev);

	std::vector<std::vector<double>> normal_matrix(_rows, std::vector<double>(_columns));

	for (int i = 0; i < _rows; i++)
	{
		for (int j = 0; j < _columns; j++)
		{
			normal_matrix[i][j] = dist(generator);
		}
	}
	
	return LinAlg::Matrix(normal_matrix);
}

LinAlg::Matrix LinAlg::Matrix::Diagonal(const std::vector<double>& _diag_values)
{
	int n = _diag_values.size();
	if (!n)
	{
		throw std::invalid_argument("[Matrix] Diagonal Matrix Build failed: empty diagonal array.");
	}

	std::vector<std::vector<double>> diagonal_matrix(n, std::vector<double>(n, 0.0));
	
	for (int i = 0; i < n; i++)
	{
		diagonal_matrix[i][i] = _diag_values[i];
	}

	return LinAlg::Matrix(diagonal_matrix);
}

// ========================================
// Matrix Utility Method(s)
// ========================================
std::pair<int, int> LinAlg::Matrix::Shape() const
{
	return this->shape;
}

int LinAlg::Matrix::Row() const
{
	return this->shape.first;
}

int LinAlg::Matrix::Column() const
{
	return this->shape.second;
}

int LinAlg::Matrix::Volume() const
{
	return this->volume;
}

bool LinAlg::Matrix::IsEmpty() const
{
	return (this->volume == 0);
}

// ========================================
// Matrix Type-Check Method(s)
// ========================================
bool LinAlg::Matrix::IsSquare() const
{
	return !this->IsEmpty() && (this->shape.first == this->shape.second);
}

bool LinAlg::Matrix::IsDiagonal(const double& _tolerance) const
{
	if (this->IsEmpty() || !this->IsSquare())
	{
		return false;
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			if (i != j && std::abs(this->data[i][j]) > _tolerance)
			{
			    return false;
			}
		}
	}
	return true;
}

bool LinAlg::Matrix::IsUpperTriangular(const double& _tolerance) const
{
	if (this->IsEmpty() || !this->IsSquare())
	{
		return false;
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < i; j++)
		{
			if (std::abs(this->data[i][j]) > _tolerance)
			{
				return false;
			}
		}
	}
	return true;
}

bool LinAlg::Matrix::IsLowerTriangular(const double& _tolerance) const
{
	if (this->IsEmpty() || !this->IsSquare())
	{
		return false;
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = (i + 1); j < this->shape.second; j++)
		{
			if (std::abs(this->data[i][j]) > _tolerance)
			{
				return false;
			}
		}
	}
	return true;
}

bool LinAlg::Matrix::IsSymmetric(const double& _tolerance) const
{
	if (this->IsEmpty() || !this->IsSquare())
	{
		return false;
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < i; j++)
		{
			if (std::abs(this->data[i][j] - this->data[j][i]) > _tolerance)
			{
				return false;
			}
		}
	}
	return true;
}

bool LinAlg::Matrix::IsSkewSymmetric(const double& _tolerance) const
{
	if (this->IsEmpty() || !this->IsSquare())
	{
		return false;
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < i; j++)
		{
			if (std::abs(this->data[i][j] + this->data[j][i]) > _tolerance)
			{
				return false;
			}
		}
	}
	return true;
}

bool LinAlg::Matrix::IsOrthogonal() const
{
	if (this->IsEmpty() || !this->IsSquare())
	{
		return false;
	}

	// Optimize this later:
	LinAlg::Matrix result = this->DotProduct(this->Transpose());

	if (result == LinAlg::Matrix::Identity(this->shape.first))
	{
		return true;
	}
	return false;
}

bool LinAlg::Matrix::IsSingular(const double& _tolerance) const
{
	if (abs(this->Determinant()) < _tolerance)
	{
		return true;
	}
	return false;
}

bool LinAlg::Matrix::IsIdempotent() const
{
	if (this->IsEmpty() || !this->IsSquare())
	{
		return false;
	}

	LinAlg::Matrix result = this->DotProduct(*this);

	if (result == *this)
	{
		return true;
	}
	return false;
}

bool LinAlg::Matrix::IsNilpotent(const int& max_power, const double& _tolerance) const
{
	// Implement later.
	return false;
}

bool LinAlg::Matrix::IsInvolutory(const double& _tolerance) const
{
	// Implement later.
	return false;
}

// ========================================
// Matrix Assignment Operator (Deep Copy)
// ========================================
void LinAlg::Matrix::operator=(const LinAlg::Matrix& _matrix)
{
	this->shape = _matrix.shape;
	this->volume = _matrix.volume;
	this->data = _matrix.data;
}

// ========================================
// Matrix Comparison Operator(s)
// ========================================
bool LinAlg::Matrix::operator==(const LinAlg::Matrix& _matrix) const
{
	if (this->shape != _matrix.shape)
	{
		return false;
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			if (std::abs(this->data[i][j] - _matrix.data[i][j]) > LinAlg::Matrix::TOLERANCE)
			{
				return false;
			}
		}
	}
	return true;
}

bool LinAlg::Matrix::operator!=(const LinAlg::Matrix& _matrix) const
{
	return !(*this == _matrix);
}

// ========================================
// Matrix Arithmetic Operator(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::operator+(const double& _scalar) const
{
	if (!Utils::IsValidData<double>({ _scalar }))
	{
		throw std::invalid_argument("[Matrix] Addition failed: invalid value.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < result.shape.first; i++)
	{
		for (int j = 0; j < result.shape.second; j++)
		{
			result.data[i][j] += _scalar;
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator-(const double& _scalar) const
{
	if (!Utils::IsValidData<double>({ _scalar }))
	{
		throw std::invalid_argument("[Matrix] Subtraction failed: invalid value.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < result.shape.first; i++)
	{
		for (int j = 0; j < result.shape.second; j++)
		{
			result.data[i][j] -= _scalar;
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator*(const double& _scalar) const
{
	if (!Utils::IsValidData<double>({ _scalar }))
	{
		throw std::invalid_argument("[Matrix] Multiplication (Hadamard) failed: invalid value.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < result.shape.first; i++)
	{
		for (int j = 0; j < result.shape.second; j++)
		{
			result.data[i][j] *= _scalar;
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator/(const double& _scalar) const
{
	if (!Utils::IsValidData<double>({ _scalar }))
	{
		throw std::invalid_argument("[Matrix] Division failed: invalid value.");
	}

	if (std::abs(_scalar) < LinAlg::Matrix::TOLERANCE)
	{
		throw std::domain_error("[Matrix] Division failed: division by near zero value detected.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < result.shape.first; i++)
	{
		for (int j = 0; j < result.shape.second; j++)
		{
			result.data[i][j] /= _scalar;
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator+(const std::vector<double>& _vector) const
{
	if (_vector.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Addition failed: column-size mismatch with input vector size.");
	}

	if (!Utils::IsValidData<double>(_vector))
	{
		throw std::invalid_argument("[Matrix] Addition failed: invalid vector value(s).");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < result.shape.first; i++)
	{
		for (int j = 0; j < result.shape.second; j++)
		{
			result.data[i][j] += _vector[j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator-(const std::vector<double>& _vector) const
{
	if (_vector.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Subtraction failed: column-size mismatch with input vector size.");
	}

	if (!Utils::IsValidData<double>(_vector))
	{
		throw std::invalid_argument("[Matrix] Subtraction failed: invalid vector value(s).");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < result.shape.first; i++)
	{
		for (int j = 0; j < result.shape.second; j++)
		{
			result.data[i][j] -= _vector[j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator*(const std::vector<double>& _vector) const
{
	if (_vector.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Multiplication failed: column-size mismatch with input vector size.");
	}

	if (!Utils::IsValidData<double>(_vector))
	{
		throw std::invalid_argument("[Matrix] Multiplication failed: invalid vector value(s).");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < result.shape.first; i++)
	{
		for (int j = 0; j < result.shape.second; j++)
		{
			result.data[i][j] *= _vector[j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator/(const std::vector<double>& _vector) const
{
	if (_vector.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Division failed: column-size mismatch with input vector size.");
	}

	if (!Utils::IsValidData<double>(_vector))
	{
		throw std::invalid_argument("[Matrix] Division failed: invalid vector value(s).");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < result.shape.first; i++)
	{
		for (int j = 0; j < result.shape.second; j++)
		{
			if (std::abs(_vector[j]) < LinAlg::Matrix::TOLERANCE)
			{
				throw std::domain_error("[Matrix] Division failed: division by near zero value detected.");
			}
			result.data[i][j] /= _vector[j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator+(const LinAlg::Matrix& _matrix) const
{
	if (this->shape.first != _matrix.shape.first || this->shape.second != _matrix.shape.second)
	{
		throw std::invalid_argument("[Matrix] Addition failed: shape mismatch with input Matrix.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			result.data[i][j] += _matrix.data[i][j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator-(const LinAlg::Matrix& _matrix) const
{
	if (this->shape.first != _matrix.shape.first || this->shape.second != _matrix.shape.second)
	{
		throw std::invalid_argument("[Matrix] Subtraction failed: shape mismatch with input Matrix.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			result.data[i][j] -= _matrix.data[i][j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator*(const LinAlg::Matrix& _matrix) const
{
	if (this->shape.first != _matrix.shape.first || this->shape.second != _matrix.shape.second)
	{
		throw std::invalid_argument("[Matrix] Multiplication failed: shape mismatch with input Matrix.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			result.data[i][j] *= _matrix.data[i][j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::operator/(const LinAlg::Matrix& _matrix) const
{
	if (this->shape.first != _matrix.shape.first || this->shape.second != _matrix.shape.second)
	{
		throw std::invalid_argument("[Matrix] Division failed: shape mismatch with input Matrix.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			if (std::abs(_matrix.data[i][j]) < LinAlg::Matrix::TOLERANCE)
			{
				throw std::domain_error("[Matrix] Division failed: division by near zero value detected.");
			}
			result.data[i][j] /= _matrix.data[i][j];
		}
	}

	return result;
}

// ========================================
// Matrix (Inplace) Arithmetic Operator(s)
// ========================================
void LinAlg::Matrix::operator+=(const double& _scalar)
{
	if (!Utils::IsValidData<double>({ _scalar }))
	{
		throw std::invalid_argument("[Matrix] Addition failed: invalid value.");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] += _scalar;
		}
	}
}

void LinAlg::Matrix::operator-=(const double& _scalar)
{
	if (!Utils::IsValidData<double>({ _scalar }))
	{
		throw std::invalid_argument("[Matrix] Subtraction failed: invalid value.");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] -= _scalar;
		}
	}
}

void LinAlg::Matrix::operator*=(const double& _scalar)
{
	if (!Utils::IsValidData<double>({ _scalar }))
	{
		throw std::invalid_argument("[Matrix] Multiplication (Hadamard) failed: invalid value.");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] *= _scalar;
		}
	}
}

void LinAlg::Matrix::operator/=(const double& _scalar)
{
	if (!Utils::IsValidData<double>({ _scalar }))
	{
		throw std::invalid_argument("[Matrix] Division failed: invalid value.");
	}

	if (std::abs(_scalar) < LinAlg::Matrix::TOLERANCE)
	{
		throw std::domain_error("[Matrix] Division failed: division by near zero value detected.");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] /= _scalar;
		}
	}
}

void LinAlg::Matrix::operator+=(const std::vector<double>& _vector)
{
	if (_vector.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Addition failed: column-size mismatch with input vector size.");
	}

	if (!Utils::IsValidData<double>(_vector))
	{
		throw std::invalid_argument("[Matrix] Addition failed: invalid vector value(s).");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] += _vector[j];
		}
	}
}

void LinAlg::Matrix::operator-=(const std::vector<double>& _vector)
{
	if (_vector.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Subtraction failed: column-size mismatch with input vector size.");
	}

	if (!Utils::IsValidData<double>(_vector))
	{
		throw std::invalid_argument("[Matrix] Subtraction failed: invalid vector value(s).");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] -= _vector[j];
		}
	}
}

void LinAlg::Matrix::operator*=(const std::vector<double>& _vector)
{
	if (_vector.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Multiplication failed: column-size mismatch with input vector size.");
	}

	if (!Utils::IsValidData<double>(_vector))
	{
		throw std::invalid_argument("[Matrix] Multiplication failed: invalid vector value(s).");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] *= _vector[j];
		}
	}
}

void LinAlg::Matrix::operator/=(const std::vector<double>& _vector)
{
	if (_vector.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Division failed: column-size mismatch with input vector size.");
	}

	if (!Utils::IsValidData<double>(_vector))
	{
		throw std::invalid_argument("[Matrix] Division failed: invalid vector value(s).");
	}

	for (const double& value : _vector)
	{
		if (std::abs(value) < LinAlg::Matrix::TOLERANCE)
		{
			throw std::domain_error("[Matrix] Division failed: division by near zero value detected.");
		}
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] /= _vector[j];
		}
	}
}

void LinAlg::Matrix::operator+=(const LinAlg::Matrix& _matrix)
{
	if (this->shape.first != _matrix.shape.first || this->shape.second != _matrix.shape.second)
	{
		throw std::invalid_argument("[Matrix] Addition failed: shape mismatch with input Matrix.");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] += _matrix.data[i][j];
		}
	}
}

void LinAlg::Matrix::operator-=(const LinAlg::Matrix& _matrix)
{
	if (this->shape.first != _matrix.shape.first || this->shape.second != _matrix.shape.second)
	{
		throw std::invalid_argument("[Matrix] Subtraction failed: shape mismatch with input Matrix.");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] -= _matrix.data[i][j];
		}
	}
}

void LinAlg::Matrix::operator*=(const LinAlg::Matrix& _matrix)
{
	if (this->shape.first != _matrix.shape.first || this->shape.second != _matrix.shape.second)
	{
		throw std::invalid_argument("[Matrix] Multiplication failed: shape mismatch with input Matrix.");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] *= _matrix.data[i][j];
		}
	}
}

void LinAlg::Matrix::operator/=(const LinAlg::Matrix& _matrix)
{
	if (this->shape.first != _matrix.shape.first || this->shape.second != _matrix.shape.second)
	{
		throw std::invalid_argument("[Matrix] Division failed: shape mismatch with input Matrix.");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			if (std::abs(_matrix.data[i][j]) < LinAlg::Matrix::TOLERANCE)
			{
				throw std::domain_error("[Matrix] Division failed: division by near zero value detected.");
			}
		}
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			this->data[i][j] /= _matrix.data[i][j];
		}
	}
}

// ========================================
// Matrix Columnwise Arithmetic Method(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::AddColumnwise(const std::vector<double>& _vector) const
{
	if (_vector.size() != this->shape.first)
	{
		throw std::invalid_argument("[Matrix] Columnwise Addition failed: row-size mismatch with input vector size.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < this->shape.second; i++)
	{
		for (int j = 0; j < this->shape.first; j++)
		{
			result.data[j][i] += _vector[j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::SubtractColumnwise(const std::vector<double>& _vector) const
{
	if (_vector.size() != this->shape.first)
	{
		throw std::invalid_argument("[Matrix] Columnwise Subtraction failed: row-size mismatch with input vector size.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < this->shape.second; i++)
	{
		for (int j = 0; j < this->shape.first; j++)
		{
			result.data[j][i] -= _vector[j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::MultiplyColumnwise(const std::vector<double>& _vector) const
{
	if (_vector.size() != this->shape.first)
	{
		throw std::invalid_argument("[Matrix] Columnwise Multiplication failed: row-size mismatch with input vector size.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < this->shape.second; i++)
	{
		for (int j = 0; j < this->shape.first; j++)
		{
			result.data[j][i] *= _vector[j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::DivideColumnwise(const std::vector<double>& _vector) const
{
	if (_vector.size() != this->shape.first)
	{
		throw std::invalid_argument("[Matrix] Columnwise Division failed: row-size mismatch with input vector size.");
	}

	LinAlg::Matrix result = *this;

	for (int i = 0; i < this->shape.second; i++)
	{
		for (int j = 0; j < this->shape.first; j++)
		{
			if (std::abs(_vector[j]) < LinAlg::Matrix::TOLERANCE)
			{
				throw std::domain_error("[Matrix] Division failed: division by near zero value detected.");
			}
			result.data[j][i] /= _vector[j];
		}
	}

	return result;
}

// ========================================
// Matrix Multiplication Method(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::DotProduct(const std::vector<std::vector<double>>& _matrix) const
{
	if (_matrix.empty() || !Utils::IsRectangular(_matrix))
	{
		throw std::runtime_error("[Matrix] Matrix Multiplication failed: input matrix is invalid.");
	}

	int rows = _matrix.size();
	int columns = _matrix[0].size();

	if (rows != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Matrix Multiplication failed: row number of input matrix mismatch with total columns of Matrix.");
	}

	std::vector<std::vector<double>> result(this->shape.first, std::vector<double>(columns, 0.0));

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			for (int k = 0; k < rows; k++)
			{
				result[i][j] += (this->data[i][k] * _matrix[k][j]);
			}
		}
	}

	return LinAlg::Matrix(result);
}

LinAlg::Matrix LinAlg::Matrix::DotProduct(const LinAlg::Matrix& _matrix) const
{
	return this->DotProduct(_matrix.data);
}

// ========================================
// Matrix Trasnpose Method
// ========================================
LinAlg::Matrix LinAlg::Matrix::Transpose() const
{
	if (this->IsEmpty())
	{
		return LinAlg::Matrix();
	}

	std::vector<std::vector<double>> transposed_data(this->shape.second, std::vector<double>(this->shape.first));

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			transposed_data[j][i] = this->data[i][j];
		}
	}

	return LinAlg::Matrix(transposed_data);
}

// ========================================
// Matrix Inverse Method(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::Inverse() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Matrix Inversion failed: empty Matrix.");
	}

	if (!this->IsSquare())
	{
		throw std::runtime_error("[Matrix] Matrix Inversion failed: matrix must be square.");
	}

	LinAlg::EliminationResult rref = this->GaussJordanElimination(Matrix::Identity(this->shape.first));

	if (rref.rank < this->shape.first)
	{
		throw std::runtime_error("[Matrix] Matrix Inversion failed: matrix is singular (not full rank).");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		if (std::abs(rref.A.data[i][i] - 1.0) > LinAlg::Matrix::TOLERANCE)
		{
			throw std::runtime_error("[Matrix] Matrix Inversion failed: RREF did not produce identity.");
		}

		for (int j = 0; j < i; j++)
		{
			if (std::abs(rref.A.data[i][j]) > LinAlg::Matrix::TOLERANCE ||
				std::abs(rref.A.data[j][i]) > LinAlg::Matrix::TOLERANCE)
			{
				throw std::runtime_error("[Matrix] Matrix Inversion failed: RREF did not produce identity.");
			}
		}
	}

	return rref.B;
}

LinAlg::Matrix LinAlg::Matrix::PseudoInverse() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Pseudoinverse failed: empty matrix.");
	}

	int m = this->shape.first;
	int n = this->shape.second;
	int rank = this->Rank();

	if (m == n && rank == n)
	{
		return this->Inverse();
	}

	if (rank == n && m >= n)
	{
		LinAlg::Matrix At = this->Transpose();
		LinAlg::Matrix AtA = At.DotProduct(*this);
		LinAlg::Matrix AtA_inv = AtA.Inverse();
		return AtA_inv.DotProduct(At);
	}

	if (rank == m && n >= m)
	{
		LinAlg::Matrix At = this->Transpose();
		LinAlg::Matrix AAt = this->DotProduct(At);
		LinAlg::Matrix AAt_inv = AAt.Inverse();
		return At.DotProduct(AAt_inv);
	}

	// Case 4: Rank-deficient - requires SVD
	throw std::runtime_error(
		"[Matrix] Pseudoinverse failed: matrix is rank-deficient (rank=" +
		std::to_string(rank) + ", shape=(" + std::to_string(m) + "," +
		std::to_string(n) + ")). Please implement SVD first, then use " +
		"SVD-based pseudoinverse for general cases."
	);
}

// ========================================
// Matrix Row-Elimination Method(s)
// ========================================
LinAlg::EliminationResult LinAlg::Matrix::GaussianElimination(const LinAlg::Matrix& _aug_matrix) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Gaussian Elimination failed: empty Matrix.");
	}

	if (!_aug_matrix.IsEmpty() && this->shape.first != _aug_matrix.shape.first)
	{
		throw std::invalid_argument("[Matrix] Gaussian Elimination failed: mismatch between no. of rows in current and augmented Matrix.");
	}

	LinAlg::Matrix aug_matrix = _aug_matrix;
	if (_aug_matrix.IsEmpty())
	{
		aug_matrix = LinAlg::Matrix({ this->shape.first, 1 }, 0.0);
	}

	int rows = this->shape.first;
	int columns = this->shape.second;

	LinAlg::Matrix coeff_matrix = *this;
	int swap_count = 0;
	int rank = 0;

	int pivot_col = 0;

	for (int pivot_row = 0; pivot_row < rows && pivot_col < columns; pivot_col++)
	{
		int max_row = pivot_row;
		double max_val = std::abs(coeff_matrix.data[pivot_row][pivot_col]);

		for (int r = pivot_row + 1; r < rows; r++)
		{
			double val = std::abs(coeff_matrix.data[r][pivot_col]);
			if (val > max_val)
			{
				max_val = val;
				max_row = r;
			}
		}

		if (max_val < LinAlg::Matrix::TOLERANCE)
		{
			for (int r = pivot_row; r < rows; r++)
			{
				coeff_matrix.data[r][pivot_col] = 0.0;
			}
			continue;
		}

		if (max_row != pivot_row)
		{
			coeff_matrix.SwapRows(max_row, pivot_row);
			aug_matrix.SwapRows(max_row, pivot_row);
			swap_count++;
		}

		for (int k = pivot_row + 1; k < rows; k++)
		{
			double factor = coeff_matrix.data[k][pivot_col] / coeff_matrix.data[pivot_row][pivot_col];

			if (std::abs(factor) < LinAlg::Matrix::TOLERANCE)
			{
				coeff_matrix.data[k][pivot_col] = 0.0;
				continue;
			}

			for (int c = pivot_col; c < columns; c++)
			{
				coeff_matrix.data[k][c] -= factor * coeff_matrix.data[pivot_row][c];

				if (std::abs(coeff_matrix.data[k][c]) < LinAlg::Matrix::TOLERANCE)
				{
					coeff_matrix.data[k][c] = 0.0;
				}
			}

			for (int c = 0; c < columns; c++)
			{
				aug_matrix.data[k][c] -= factor * aug_matrix.data[pivot_row][c];

				if (std::abs(aug_matrix.data[k][c]) < LinAlg::Matrix::TOLERANCE)
				{
					aug_matrix.data[k][c] = 0.0;
				}
			}
		}

		rank++;
		pivot_row++;
	}

	return LinAlg::EliminationResult(coeff_matrix, aug_matrix, rank, swap_count);
}

LinAlg::EliminationResult LinAlg::Matrix::GaussJordanElimination(const LinAlg::Matrix& _aug_matrix) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Gauss-Jordan Elimination failed: empty Matrix.");
	}

	LinAlg::EliminationResult ref = this->GaussianElimination(_aug_matrix);

	int i = ref.rank - 1;
	int pivot = ref.A.shape.second - 1;

	for (int i = ref.rank - 1; i >= 0; i--)
	{
		int pivot = -1;
		for (int j = 0; j < ref.A.shape.second; j++)
		{
			if (std::abs(ref.A.data[i][j]) > LinAlg::Matrix::TOLERANCE)
			{
				pivot = j;
				break;
			}
		}

		if (pivot == -1)
		{
			continue;
		}

		double pivot_value = ref.A.data[i][pivot];

		for (int c = pivot; c < ref.A.shape.second; c++)
		{
			ref.A.data[i][c] /= pivot_value;
		}

		for (int c = 0; c < ref.B.shape.second; c++)
		{
			ref.B.data[i][c] /= pivot_value;
		}

		for (int r = 0; r < i; r++)
		{
			double factor = ref.A.data[r][pivot];

			if (std::abs(factor) < LinAlg::Matrix::TOLERANCE)
			{
				continue;
			}

			for (int c = pivot; c < ref.A.shape.second; c++)
			{
				ref.A.data[r][c] -= (factor * ref.A.data[i][c]);
			}

			for (int c = 0; c < ref.B.shape.second; c++)
			{
				ref.B.data[r][c] -= (factor * ref.B.data[i][c]);
			}
		}
	}

	return ref;
}

// ========================================
// Matrix Property Method(s)
// ========================================
double LinAlg::Matrix::Determinant() const
{
	if (!this->IsSquare())
	{
		throw std::runtime_error("[Matrix] Determinant Computation failed: determinant is not defined for non-square matrix.");
	}

	LinAlg::EliminationResult row_echelon_form = this->GaussianElimination();

	if (row_echelon_form.rank < this->shape.first)
	{
		return 0.0;
	}

	double det = (row_echelon_form.swapCount % 2) ? -1.0 : 1.0;

	for (int i = 0; i < row_echelon_form.A.shape.first; i++)
	{
		det *= row_echelon_form.A.data[i][i];
	}

	return det;
}

double LinAlg::Matrix::Trace() const
{
	if (!this->IsSquare())
	{
		throw std::runtime_error("[Matrix] Trace Computation failed: trace is not defined for non-square matrix.");
	}

	double trace = 0.0;
	for (int i = 0; i < this->shape.first; i++)
	{
		trace += this->data[i][i];
	}

	return trace;
}

int LinAlg::Matrix::Rank() const
{
	LinAlg::EliminationResult row_echelon_form = this->GaussianElimination();
	return row_echelon_form.rank;
}

// ========================================
// Matrix Statistical Method(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::ReduceSum(const bool& _row_wise) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Ruduce Sum failed: empty Matrix.");
	}

	int n = _row_wise ? this->shape.first : this->shape.second;
	std::vector<double> reduced_sum(n, 0.0);

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			if (_row_wise)
			{
				reduced_sum[i] += this->data[i][j];
			}
			else
			{
				reduced_sum[j] += this->data[i][j];
			}
		}
	}

	std::pair<int, int> p = (_row_wise) ? std::make_pair(n, 1) : std::make_pair(1, n);
	return LinAlg::Matrix(p, reduced_sum);
}

LinAlg::Matrix LinAlg::Matrix::ReduceMean(const bool& _row_wise) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Ruduce Mean failed: empty Matrix.");
	}

	int n = _row_wise ? this->shape.second : this->shape.first;

	LinAlg::Matrix reduced_sum = this->ReduceSum(_row_wise);
	LinAlg::Matrix reduced_mean = reduced_sum / n;

	return reduced_mean;
}

LinAlg::Matrix LinAlg::Matrix::ReduceVar(const bool& _row_wise, const bool& _inference) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Ruduce Variance failed: empty Matrix.");
	}

	int n = _row_wise ? this->shape.second : this->shape.first;

	LinAlg::Matrix reduced_mean = this->ReduceMean(_row_wise);
	LinAlg::Matrix reduced_var(reduced_mean.shape, 0.0);

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			if (_row_wise)
			{
				double diff = this->data[i][j] - reduced_mean.data[i][0];
				reduced_var.data[i][0] += (diff * diff);
			}
			else
			{
				double diff = this->data[i][j] - reduced_mean.data[0][j];
				reduced_var.data[0][j] += (diff * diff);
			}
		}
	}

	n -= (_inference && n > 1) ? 1 : 0;
	reduced_var = reduced_var / n;

	return reduced_var;
}

LinAlg::Matrix LinAlg::Matrix::ReduceMax(const bool& _row_wise) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Ruduce Max failed: empty Matrix.");
	}

	int n = _row_wise ? this->shape.first : this->shape.second;
	std::vector<double> reduced_max(n, std::numeric_limits<double>::lowest());

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			if (_row_wise)
			{
				reduced_max[i] = std::max(reduced_max[i], this->data[i][j]);
			}
			else
			{
				reduced_max[j] = std::max(reduced_max[j], this->data[i][j]);
			}
		}
	}

	std::pair<int, int> p = (_row_wise) ? std::make_pair(n, 1) : std::make_pair(1, n);
	return LinAlg::Matrix(p, reduced_max);
}

LinAlg::Matrix LinAlg::Matrix::ReduceMin(const bool& _row_wise) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Ruduce Min failed: empty Matrix.");
	}

	int n = _row_wise ? this->shape.first : this->shape.second;
	std::vector<double> reduced_min(n, std::numeric_limits<double>::max());

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			if (_row_wise)
			{
				reduced_min[i] = std::min(reduced_min[i], this->data[i][j]);
			}
			else
			{
				reduced_min[j] = std::min(reduced_min[j], this->data[i][j]);
			}
		}
	}

	std::pair<int, int> p = (_row_wise) ? std::make_pair(n, 1) : std::make_pair(1, n);
	return LinAlg::Matrix(p, reduced_min);
}

double LinAlg::Matrix::Sum() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Compute Sum failed: empty Matrix.");
	}

	double sum = 0.0;
	for (const auto& row : this->data)
	{
		for (const double& value : row)
		{
			sum += value;
		}
	}

	return sum;
}

double LinAlg::Matrix::Mean() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Compute Mean failed: empty Matrix.");
	}

	double sum = this->Sum();
	return sum / this->volume;
}

double LinAlg::Matrix::Var(const bool& _inference) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Compute Variance failed: empty Matrix.");
	}

	double mean = this->Mean();
	double var = 0.0;

	for (const auto& row : this->data)
	{
		for (const double& value : row)
		{
			double diff = value - mean;
			var += (diff * diff);
		}
	}

	int n = (_inference && this->volume > 1) ? this->volume - 1 : this->volume;
	return var / n;
}

double LinAlg::Matrix::Max() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Compute Max failed: empty Matrix.");
	}

	double max_value = this->data[0][0];
	for (const auto& row : this->data)
	{
		for (const double& value : row)
		{
			max_value = std::max(max_value, value);
		}
	}

	return max_value;
}

double LinAlg::Matrix::Min() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Compute Min failed: empty Matrix.");
	}

	double min_value = this->data[0][0];
	for (const auto& row : this->data)
	{
		for (const double& value : row)
		{
			min_value = std::min(min_value, value);
		}
	}

	return min_value;
}

// ========================================
// Matrix Reshape Method
// ========================================
LinAlg::Matrix LinAlg::Matrix::Reshape(const std::pair<int, int> _shape) const
{
	if (_shape.first <= 0 || _shape.second <= 0)
	{
		throw std::invalid_argument("[Matrix] Reshaping Matrix failed: found negative value of rows and/or columns.");
	}

	if ((_shape.first * _shape.second) != this->volume)
	{
		throw std::invalid_argument("[Matrix] Reshaping Matrix failed: shape-volume mismatch with Matrix volume.");
	}

	std::vector<std::vector<double>> reshaped_data(_shape.first, std::vector<double>(_shape.second));

	int r = 0, c = 0;
	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			reshaped_data[r][c] = this->data[i][j];

			c++;
			if (c >= _shape.second)
			{
				r++;
				c = 0;
			}
		}
	}

	return LinAlg::Matrix(reshaped_data);
}

// ========================================
// Matrix Row & Column Swap Method(s)
// ========================================
void LinAlg::Matrix::SwapRows(const int& _row_1, const int& _row_2)
{
	if (_row_1 < 0 || _row_1 >= this->shape.first)
	{
		throw std::out_of_range("[Matrix] Swap Rows failed: first row-number is out of bounds.");
	}

	if (_row_2 < 0 || _row_2 >= this->shape.first)
	{
		throw std::out_of_range("[Matrix] Swap Rows failed: second row-number is out of bounds.");
	}

	for (int i = 0; i < this->shape.second; i++)
	{
		double temp = this->data[_row_1][i];
		this->data[_row_1][i] = this->data[_row_2][i];
		this->data[_row_2][i] = temp;
	}
}

void LinAlg::Matrix::SwapColumns(const int& _col_1, const int& _col_2)
{
	if (_col_1 < 0 || _col_1 >= this->shape.second)
	{
		throw std::out_of_range("[Matrix] Swap Columns failed: first column-number is out of bounds.");
	}

	if (_col_2 < 0 || _col_2 >= this->shape.second)
	{
		throw std::out_of_range("[Matrix] Swap Columns failed: second column-number is out of bounds.");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		double temp = this->data[i][_col_1];
		this->data[i][_col_1] = this->data[i][_col_2];
		this->data[i][_col_2] = temp;
	}
}

// ========================================
// Matrix Accessing & Indexing Method(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::Submatrix(const std::pair<int, int> _start, const std::pair<int, int> _end) const
{
	if (_start.first < 0 || _start.second < 0 || _start.first >= this->shape.first || _start.second >= this->shape.second)
	{
		throw std::out_of_range("[Matrix] Sub-Matrix Create failed: start index(row, column) is out of bounds.");
	}

	if (_end.first < 0 || _end.second < 0 || _end.first > this->shape.first || _end.second > this->shape.second)
	{
		throw std::out_of_range("[Matrix] Sub-Matrix Create failed: end index(row, column) is out of bounds.");
	}

	if (_start.first >= _end.first)
	{
		throw std::invalid_argument("[Matrix] Sub-Matrix Create failed: start row has greater/equal value than end row.");
	}

	if (_start.second >= _end.second)
	{
		throw std::invalid_argument("[Matrix] Sub-Matrix Create failed: start column has greater/equal value than end column.");
	}

	int rows = _end.first - _start.first;
	int columns = _end.second - _start.second;

	std::vector<std::vector<double>> sub_matrix(rows, std::vector<double>(columns));

	for (int i = _start.first, r = 0; i < _end.first; i++, r++)
	{
		for (int j = _start.second, c = 0; j < _end.second; j++, c++)
		{
			sub_matrix[r][c] = this->data[i][j];
		}
	}

	return LinAlg::Matrix(sub_matrix);
}

std::vector<double> LinAlg::Matrix::GetRow(const int& _row_index) const
{
	if (_row_index < 0 || _row_index >= this->shape.first)
	{
		throw std::out_of_range("[Matrix] Get Row failed: row index is out of bounds.");
	}

	return this->data[_row_index];
}

std::vector<double> LinAlg::Matrix::GetColumn(const int& _column_index) const
{
	if (_column_index < 0 || _column_index >= this->shape.second)
	{
		throw std::out_of_range("[Matrix] Get Column failed: column index is out of bounds.");
	}

	std::vector<double> column_data(this->shape.first);
	
	for (int i = 0; i < this->shape.first; i++)
	{
		column_data[i] = this->data[i][_column_index];
	}

	return column_data;
}

LinAlg::Matrix LinAlg::Matrix::GetRows(const std::vector<int>& _row_indices) const
{
	if (_row_indices.empty())
	{
		return LinAlg::Matrix();
	}

	std::vector<bool> row_filter(this->shape.first, false);

	for (const int& index : _row_indices)
	{
		if (index < 0 || index >= this->shape.first)
		{
			throw std::out_of_range("[Matrix] Get Rows failed: row index is out of bounds.");
		}
		row_filter[index] = true;
	}

	std::vector<std::vector<double>> sub_matrix;

	for (int i = 0; i < this->shape.first; i++)
	{
		if (row_filter[i])
		{
			sub_matrix.push_back(this->GetRow(i));
		}
	}

	return LinAlg::Matrix(sub_matrix);
}

LinAlg::Matrix LinAlg::Matrix::GetColumns(const std::vector<int>& _column_indices) const
{
	if (_column_indices.empty())
	{
		return LinAlg::Matrix();
	}

	std::vector<bool> column_filter(this->shape.second, false);

	for (const int& index : _column_indices)
	{
		if (index < 0 || index >= this->shape.second)
		{
			throw std::out_of_range("[Matrix] Get Columns failed: column index is out of bounds.");
		}
		column_filter[index] = true;
	}

	std::vector<std::vector<double>> sub_matrix;

	for (int i = 0; i < this->shape.second; i++)
	{
		if (column_filter[i])
		{
			sub_matrix.push_back(this->GetColumn(i));
		}
	}

	return LinAlg::Matrix(sub_matrix);
}

// ========================================
// Matrix Print Method
// ========================================
void LinAlg::Matrix::Print() const
{
	for (const auto& row : this->data)
	{
		for (const double& value : row)
		{
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}
}

// ========================================
// Matrix Norm Computation Method(s)
// ========================================
double LinAlg::Matrix::FrobeniusNorm() const
{
	return 0.0;
}

double LinAlg::Matrix::SpectralNorm() const
{
	return 0.0;
}

double LinAlg::Matrix::NuclearNorm() const
{
	return 0.0;
}

double LinAlg::Matrix::InfinityNorm() const
{
	return 0.0;
}

double LinAlg::Matrix::OneNorm() const
{
	return 0.0;
}

// ========================================
// Matrix Decomposition Method(s)
// ========================================
LinAlg::LUResult LinAlg::Matrix::LUDecomposition() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] LU Decomposition failed: empty Matrix.");
	}

	int rows = this->shape.first;
	int columns = this->shape.second;

	LinAlg::Matrix L = LinAlg::Matrix::Identity(rows);
	LinAlg::Matrix U = *this;

	std::vector<int> permutation(rows);
	std::iota(permutation.begin(), permutation.end(), 0);

	int pivot_col = 0;

	for (int pivot_row = 0; pivot_row < rows && pivot_col < columns; pivot_col++)
	{
		int max_row = pivot_row;
		double max_val = std::abs(U.data[pivot_row][pivot_col]);

		for (int r = pivot_row + 1; r < rows; r++)
		{
			double val = std::abs(U.data[r][pivot_col]);
			if (val > max_val)
			{
				max_val = val;
				max_row = r;
			}
		}

		if (max_val < LinAlg::Matrix::TOLERANCE)
		{
			for (int r = pivot_row; r < rows; r++)
			{
				U.data[r][pivot_col] = 0.0;
			}
			continue;
		}

		if (max_row != pivot_row)
		{
			U.SwapRows(max_row, pivot_row);

			for (int j = 0; j < pivot_col; j++)
			{
				std::swap(L.data[max_row][j], L.data[pivot_row][j]);
			}

			std::swap(permutation[max_row], permutation[pivot_row]);
		}

		for (int k = pivot_row + 1; k < rows; k++)
		{
			double factor = U.data[k][pivot_col] / U.data[pivot_row][pivot_col];

			if (std::abs(factor) < LinAlg::Matrix::TOLERANCE)
			{
				U.data[k][pivot_col] = 0.0;
				continue;
			}

			L.data[k][pivot_col] = factor;

			for (int c = pivot_col; c < columns; c++)
			{
				U.data[k][c] -= factor * U.data[pivot_row][c];

				if (std::abs(U.data[k][c]) < LinAlg::Matrix::TOLERANCE)
				{
					U.data[k][c] = 0.0;
				}
			}
		}

		pivot_row++;
	}

	LinAlg::Matrix P({ rows, rows }, 0.0);
	for (int i = 0; i < rows; i++)
	{
		P.data[permutation[i]][i] = 1.0;
	}

	return LinAlg::LUResult{ L, U, P };
}

LinAlg::LDUResult LinAlg::Matrix::LDUDecomposition() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] LDU Decomposition failed: empty Matrix.");
	}

	LinAlg::LUResult result = this->LUDecomposition();

	int rows = result.U.shape.first;
	int columns = result.U.shape.second;

	LinAlg::Matrix D = LinAlg::Matrix::Identity(rows);

	int rank = std::min(rows, columns);

	for (int i = 0; i < rank; i++)
	{
		double diag_value = result.U.data[i][i];

		if (std::abs(result.U.data[i][i]) < LinAlg::Matrix::TOLERANCE)
		{
			D.data[i][i] = 0.0;
			continue;
		}

		D.data[i][i] = diag_value;

		for (int j = i; j < columns; j++)
		{
			result.U.data[i][j] /= diag_value;
		}
	}

	return LinAlg::LDUResult(result.L, D, result.U, result.P);
}

LinAlg::QRResult LinAlg::Matrix::QRDecomposition(const bool& _modified_gs) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] QR Decomposition failed: empty matrix.");
	}

	int rows = this->shape.first;
	int columns = this->shape.second;

	// donot throw error here.

	if (_modified_gs)
	{
		LinAlg::Matrix Q = *this;
		LinAlg::Matrix R = LinAlg::Matrix({ columns, columns }, 0.0);

		for (int i = 0; i < columns; i++)
		{
			for (int j = 0; j < i; j++)
			{
				R.data[j][i] = 0.0;

				for (int k = 0; k < rows; k++)
				{
					R.data[j][i] += (Q.data[k][i] * Q.data[k][j]);
				}

				for (int k = 0; k < rows; k++)
				{
					Q.data[k][i] -= (R.data[j][i] * Q.data[k][j]);
				}
			}

			double squared_norm = 0.0;

			for (int k = 0; k < rows; k++)
			{
				squared_norm += (Q.data[k][i] * Q.data[k][i]);
			}

			R.data[i][i] = std::sqrt(squared_norm);

			if (R.data[i][i] < LinAlg::Matrix::TOLERANCE)
			{
				throw std::runtime_error("[Matrix] QR Decomposition failed: linearly dependent columns at column " + std::to_string(i));
			}

			for (int k = 0; k < rows; k++)
			{
				Q.data[k][i] /= R.data[i][i];
			}
		}

		return LinAlg::QRResult(Q, R);
	}
	// will implement household later.
	return LinAlg::QRResult();
}

LinAlg::SVDResult LinAlg::Matrix::SVDDecomposition() const
{
	return LinAlg::SVDResult();
}

LinAlg::CholeskyResult LinAlg::Matrix::CholeskyDecomposition() const
{
	return LinAlg::CholeskyResult();
}

LinAlg::EigenResult LinAlg::Matrix::EigenDecomposition() const
{
	return LinAlg::EigenResult();
}

LinAlg::EigenResult LinAlg::Matrix::SpectralDecomposition() const
{
	return LinAlg::EigenResult();
}
