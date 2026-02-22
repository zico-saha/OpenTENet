#include "Matrix.h"
#include "MatrixDecompResult.h"

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
// [Private] Noise Clearing Method(s)
// ========================================
void LinAlg::Matrix::ClearNoise()
{
	for (auto& row : this->data)
	{
		for (double& value : row)
		{
			if (std::abs(value) < LinAlg::Matrix::TOLERANCE)
			{
				value = 0.0;
			}
		}
	}
}

// ========================================
// [Private] Apply Elementwise Operation Method(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::Apply(const std::function<double(double)>& _func) const
{
	LinAlg::Matrix result(this->shape, 0.0);

	for (int i = 0; i < this->shape.first; i++)
	{
		for (int j = 0; j < this->shape.second; j++)
		{
			result.data[i][j] = _func(this->data[i][j]);
		}
	}

	return result;
}

// ========================================
// [Private] Helper Matrix Method(s)
// ========================================
std::pair<double, double> LinAlg::Matrix::ComputeGivens(const double& _value_1, const double& _value_2) const
{
	if (std::abs(_value_2) < LinAlg::Matrix::TOLERANCE)
	{
		return { 1.0, 0.0 };
	}

	if (std::abs(_value_2) > std::abs(_value_1))
	{
		double tau = -(_value_1 / _value_2);
		double s = 1.0 / std::sqrt(1.0 + (tau * tau));
		double c = s * tau;
		return { c, s };
	}
	else
	{
		double tau = -(_value_2 / _value_1);
		double c = 1.0 / std::sqrt(1.0 + (tau * tau));
		double s = c * tau;
		return { c, s };
	}
}

double LinAlg::Matrix::WilkinsonShift() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] WilkinsonShift Computation failed: empty Matrix.");
	}

	if (this->shape.first != 2 || this->shape.second != 2)
	{
		throw std::runtime_error("[Matrix] WilkinsonShift Computation failed: Matrix must be of shape(2x2).");
	}

	double a = this->data[0][0];
	double b = this->data[0][1];
	double c = this->data[1][0];
	double d = this->data[1][1];

	double delta = (a - d) / 2;
	double sign = (delta >= 0) ? 1.0 : -1.0;

	double D = (delta * delta) + (b * c);
	D = (std::abs(D) < LinAlg::Matrix::TOLERANCE) ? 0.0 : D;

	if (D < 0.0)
	{
		throw std::runtime_error("[Matrix] WilkinsonShift Computation failed: Complex Eigen roots are formed.");
	}

	double mu = d - (sign * b * c) / (std::abs(delta) + std::sqrt(D));

	return mu;
}

LinAlg::Matrix LinAlg::Matrix::PartialMatMul(const LinAlg::Matrix& _sub_matrix, const std::pair<int, int>& _start, const std::pair<int, int>& _end, const bool& _left_multiply) const
{
	if (_start.first < 0 || _start.second < 0)
	{
		throw std::invalid_argument("[Matrix] Partial-MatMul failed: co-ordinate contains negative value.");
	}

	if (_start.first >= _end.first || _start.second >= _end.second)
	{
		throw std::invalid_argument("[Matrix] Partial-MatMul failed: invalid start & end matrix <row, col> pair.");
	}

	int k = (_left_multiply) ? this->shape.first : this->shape.second;

	if (_end.first > k || _end.second > k)
	{
		throw std::invalid_argument("[Matrix] Partial-MatMul failed: co-ordinate value(s) exceeds Matrix shape-bounds.");
	}

	std::pair<int, int> shape = std::make_pair((_end.first - _start.first), (_end.second - _start.second));

	if (_sub_matrix.shape != shape)
	{
		throw std::invalid_argument("[Matrix] Partial-MatMul failed: shape mismatch between sub-Matrix and co-ordinate bounds.");
	}

	LinAlg::Matrix result = *this;

	if (_left_multiply)
	{
		for (int row = _start.first, r = 0; row < _end.first && r < _sub_matrix.shape.first; row++, r++)
		{
			for (int col = 0; col < result.shape.second; col++)
			{
				double product = 0.0;
				for (int p = _start.second, q = 0; p < _end.second && q < _sub_matrix.shape.second; p++, q++)
				{
					product += (_sub_matrix.data[r][q] * this->data[p][col]);
				}
				product += (_start.second <= row && _end.second > row) ? 0.0 : this->data[row][col];

				result.data[row][col] = product;
			}
		}
	}
	else
	{
		for (int row = 0; row < result.shape.first; row++)
		{
			for (int col = _start.second, c = 0; col < _end.second && c < _sub_matrix.shape.second; col++, c++)
			{
				double product = 0.0;
				for (int p = _start.first, q = 0; p < _end.first && q < _sub_matrix.shape.first; p++, q++)
				{
					product += (this->data[row][p] * _sub_matrix.data[q][c]);
				}
				product += (_start.first <= col && _end.first > col) ? 0.0 : this->data[row][col];

				result.data[row][col] = product;
			}
		}
	}

	return result;
}

void LinAlg::Matrix::PermuteRows(const std::vector<int>& _permutation)
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Row Permutation failed: empty Matrix for permutation.");
	}

	if (static_cast<int>(_permutation.size()) != this->shape.first)
	{
		throw std::invalid_argument("[Matrix] Row Permutation failed: permutation array size mismatch with Matrix row-count.");
	}

	if (Utils::IsAnyNegative(_permutation))
	{
		throw std::invalid_argument("[Matrix] Row Permutation failed: negative value found in permutation array.");
	}

	if (!Utils::IsBounded(_permutation, this->shape.first, -1, true))
	{
		throw std::invalid_argument("[Matrix] Row Permutation failed: unbounded values found in permutation array w.r.t Matrix row-count.");
	}

	std::vector<int> points(_permutation.size());
	for (int p = 0; p < _permutation.size(); p++)
	{
		points[_permutation[p]] = p;
	}

	std::vector<double> temp = this->data[0];
	int from = 0;

	do
	{
		int to = points[from];
		std::swap(temp, this->data[to]);

		from = to;
	} while (from != 0);
}

void LinAlg::Matrix::PermuteColumns(const std::vector<int>& _permutation)
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Column Permutation failed: empty Matrix for permutation.");
	}

	if (static_cast<int>(_permutation.size()) != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Column Permutation failed: permutation array size mismatch with Matrix column-count.");
	}

	if (Utils::IsAnyNegative(_permutation))
	{
		throw std::invalid_argument("[Matrix] Column Permutation failed: negative value found in permutation array.");
	}

	if (!Utils::IsBounded(_permutation, this->shape.second, -1, true))
	{
		throw std::invalid_argument("[Matrix] Column Permutation failed: unbounded values found in permutation array w.r.t Matrix column-count.");
	}

	std::vector<int> points(_permutation.size());
	for (int p = 0; p < _permutation.size(); p++)
	{
		points[_permutation[p]] = p;
	}

	std::vector<double> temp = this->GetColumn(0);
	int from = 0;

	do
	{
		int to = points[from];
		for (int row = 0; row < this->shape.first; row++)
		{
			std::swap(temp[row], this->data[row][to]);
		}
		from = to;
	} while (from != 0);
}

// ========================================
// Matrix Constructor(s)
// ========================================
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

LinAlg::Matrix::Matrix(const std::pair<int, int>& _shape, const std::vector<double>& _data)
{
	if (_shape.first <= 0 || _shape.second <= 0)
	{
		throw std::invalid_argument("[Matrix] Constructor failed: no. of row and column of a matrix must be > 0.");
	}

	if (!Utils::IsValidData(_data))
	{
		throw std::invalid_argument("[Matrix] Constructor failed: invalid value found in data.");
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

	LinAlg::Matrix I_matrix({ _n, _n }, 0.0);

    for (int row = 0; row < _n; row++)
    {
		I_matrix.data[row][row] = _scale;
    }

    return I_matrix;
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

	LinAlg::Matrix uniform_matrix({ _rows, _columns }, 0.0);

    for (int row = 0; row < _rows; row++)
    {
        for (int col = 0; col < _columns; col++)
        {
			uniform_matrix.data[row][col] = dist(generator);
        }
    }

    return uniform_matrix;
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

	LinAlg::Matrix normal_matrix({ _rows, _columns }, 0.0);

	for (int row = 0; row < _rows; row++)
	{
		for (int col = 0; col < _columns; col++)
		{
			normal_matrix.data[row][col] = dist(generator);
		}
	}
	
	return normal_matrix;
}

LinAlg::Matrix LinAlg::Matrix::Diagonal(const std::vector<double>& _diag_values)
{
	int n = _diag_values.size();
	if (!n)
	{
		throw std::invalid_argument("[Matrix] Diagonal Matrix Build failed: empty diagonal array.");
	}

	LinAlg::Matrix diagonal_matrix({ n, n }, 0.0);
	
	for (int row = 0; row < n; row++)
	{
		diagonal_matrix.data[row][row] = _diag_values[row];
	}

	return diagonal_matrix;
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

bool LinAlg::Matrix::IsBidiagonal(const std::string& _type, const double& _tolerance) const
{
	if (this->IsEmpty())
	{
		return false;
	}

	std::string type = _type;

	std::transform(type.begin(), type.end(), type.begin(),
		[](unsigned char c) { return std::tolower(c); });

	bool is_upper_diagonal = false;
	bool is_lower_diagonal = false;

	for (int row = 0; row < this->shape.first; row++)
	{
		for (int col = 0; col < this->shape.second; col++)
		{
			bool non_zero = (std::abs(this->data[row][col]) > _tolerance);
			int d = row - col;

			if (type == "any")
			{
				if (non_zero && d != 0)
				{
					if (d == -1)
					{
						is_upper_diagonal = true;
					}
					else if (d == 1)
					{
						is_lower_diagonal = true;
					}
					else
					{
						return false;
					}
				}

				if (is_upper_diagonal && is_lower_diagonal)
				{
					return false;
				}
			}
			else if (type == "upper")
			{
				if (non_zero && d != 0 && d != -1)
				{
					return false;
				}
			}
			else if(type == "lower")
			{
				if (non_zero && d != 0 && d != 1)
				{
					return false;
				}
			}
			else
			{
				throw std::invalid_argument("[Matrix] Is Bidiagonal Check failed: got invalid type for bidiagonal Matrix check.");
			}
		}
	}

	return true;
}

bool LinAlg::Matrix::IsTridiagonal(const double& _tolerance) const
{
	if (this->IsEmpty() || !this->IsSquare())
	{
		return false;
	}

	for (int row = 0; row < this->shape.first; row++)
	{
		for (int col = 0; col < this->shape.second; col++)
		{
			int d = std::abs(row - col);
			if (std::abs(this->data[row][col]) > _tolerance && d > 1)
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
	LinAlg::Matrix result = this->MatMul(this->Transpose());

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

	LinAlg::Matrix result = this->MatMul(*this);

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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	this->ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
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

	result.ClearNoise();
	return result;
}

// ========================================
// Matrix Multiplication Method(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::MatMul(const std::vector<std::vector<double>>& _matrix) const
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

	LinAlg::Matrix result({ this->shape.first, columns }, 0.0);

	for (int row = 0; row < this->shape.first; row++)
	{
		for (int col = 0; col < columns; col++)
		{
			double res = 0.0;
			for (int k = 0; k < rows; k++)
			{
				res += (this->data[row][k] * _matrix[k][col]);
			}

			res = (std::abs(res) < LinAlg::Matrix::TOLERANCE) ? 0.0 : res;
			result.data[row][col] = res;
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::Matrix::MatMul(const LinAlg::Matrix& _matrix) const
{
	return this->MatMul(_matrix.data);
}

LinAlg::Matrix LinAlg::Matrix::MatMul(const LinAlg::Matrix& _matrix_1, const LinAlg::Matrix& _matrix_2)
{
	return _matrix_1.MatMul(_matrix_2);
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

	LinAlg::Matrix transposed_matrix({ this->shape.second, this->shape.first }, 0.0);

	for (int row = 0; row < this->shape.first; row++)
	{
		for (int col = 0; col < this->shape.second; col++)
		{
			transposed_matrix.data[col][row] = this->data[row][col];
		}
	}

	return transposed_matrix;
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
	// Build after SVD is done.
	return LinAlg::Matrix();
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

		for (int row = pivot_row + 1; row < rows; row++)
		{
			double val = std::abs(coeff_matrix.data[row][pivot_col]);
			if (val > max_val)
			{
				max_val = val;
				max_row = row;
			}
		}

		if (max_val < LinAlg::Matrix::TOLERANCE)
		{
			for (int row = pivot_row; row < rows; row++)
			{
				coeff_matrix.data[row][pivot_col] = 0.0;
			}
			continue;
		}

		if (max_row != pivot_row)
		{
			coeff_matrix.SwapRows(max_row, pivot_row);
			aug_matrix.SwapRows(max_row, pivot_row);
			swap_count++;
		}

		for (int row = pivot_row + 1; row < rows; row++)
		{
			double factor = coeff_matrix.data[row][pivot_col] / coeff_matrix.data[pivot_row][pivot_col];

			if (std::abs(factor) < LinAlg::Matrix::TOLERANCE)
			{
				coeff_matrix.data[row][pivot_col] = 0.0;
				continue;
			}

			for (int col = pivot_col; col < columns; col++)
			{
				coeff_matrix.data[row][col] -= factor * coeff_matrix.data[pivot_row][col];

				if (std::abs(coeff_matrix.data[row][col]) < LinAlg::Matrix::TOLERANCE)
				{
					coeff_matrix.data[row][col] = 0.0;
				}
			}

			for (int col = 0; col < aug_matrix.shape.second; col++)
			{
				aug_matrix.data[row][col] -= factor * aug_matrix.data[pivot_row][col];

				if (std::abs(aug_matrix.data[row][col]) < LinAlg::Matrix::TOLERANCE)
				{
					aug_matrix.data[row][col] = 0.0;
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

	int pivot = ref.A.shape.second - 1;

	for (int row = ref.rank - 1; row >= 0; row--)
	{
		int pivot = -1;
		for (int col = 0; col < ref.A.shape.second; col++)
		{
			if (std::abs(ref.A.data[row][col]) > LinAlg::Matrix::TOLERANCE)
			{
				pivot = col;
				break;
			}
		}

		if (pivot == -1)
		{
			continue;
		}

		double pivot_value = ref.A.data[row][pivot];

		for (int col = pivot; col < ref.A.shape.second; col++)
		{
			ref.A.data[row][col] /= pivot_value;
		}

		for (int col = 0; col < ref.B.shape.second; col++)
		{
			ref.B.data[row][col] /= pivot_value;
		}

		for (int r = 0; r < row; r++)
		{
			double factor = ref.A.data[r][pivot];

			if (std::abs(factor) < LinAlg::Matrix::TOLERANCE)
			{
				continue;
			}

			for (int c = pivot; c < ref.A.shape.second; c++)
			{
				ref.A.data[r][c] -= (factor * ref.A.data[row][c]);
			}

			for (int c = 0; c < ref.B.shape.second; c++)
			{
				ref.B.data[r][c] -= (factor * ref.B.data[row][c]);
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

std::vector<double> LinAlg::Matrix::Diag(const bool& _sign) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Get Diagonal failed: empty Matrix.");
	}

	if (!this->IsSquare())
	{
		throw std::runtime_error("[Matrix] Get Diagonal failed: diagonal is only defined for square Matrix.");
	}

	std::vector<double> diagonal(this->shape.first);

	for (int i = 0; i < this->shape.first; i++)
	{
		diagonal[i] = this->data[i][i];

		if (_sign)
		{
			diagonal[i] = (diagonal[i] < 0.0) ? -1.0 : 1.0;
		}
	}

	return diagonal;
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

	LinAlg::Matrix reshaped_matrix({ _shape.first, _shape.second }, 0.0);

	int r = 0, c = 0;
	for (int row = 0; row < this->shape.first; row++)
	{
		for (int col = 0; col < this->shape.second; col++)
		{
			reshaped_matrix.data[r][c] = this->data[row][col];

			c++;
			if (c >= _shape.second)
			{
				r++;
				c = 0;
			}
		}
	}

	return reshaped_matrix;
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
		std::swap(this->data[_row_1][i], this->data[_row_2][i]);
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
		std::swap(this->data[i][_col_1], this->data[i][_col_2]);
	}
}

// ========================================
// Matrix Patching Method
// ========================================
void LinAlg::Matrix::Patch(const LinAlg::Matrix& _matrix, const std::pair<int, int>& _start, const std::pair<int, int>& _end)
{
	if (_start.first < 0 || _start.second < 0)
	{
		throw std::invalid_argument("[Matrix] Patching failed: co-ordinate(s) contains negative value.");
	}

	if (_start.first >= _end.first || _start.second >= _end.second)
	{
		throw std::invalid_argument("[Matrix] Patching failed: invalid start & end matrix <row, col> pair.");
	}

	if (_end.first > this->shape.first || _end.second > this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Patching failed: co-ordinate value(s) exceeds Matrix shape-bounds.");
	}

	std::pair<int, int> shape = std::make_pair((_end.first - _start.first), (_end.second - _start.second));

	if (_matrix.shape != shape)
	{
		throw std::invalid_argument("[Matrix] Patching failed: shape mismatch between sub-Matrix and co-ordinate bounds.");
	}

	for (int row = _start.first, r = 0; row < _end.first; row++, r++)
	{
		for (int col = _start.second, c = 0; col < _end.second; col++, c++)
		{
			this->data[row][col] = _matrix.data[r][c];
		}
	}
}

// ========================================
// Matrix Getter (Accessing/Indexing) Method(s)
// ========================================
LinAlg::Matrix LinAlg::Matrix::Submatrix(const std::pair<int, int>& _start, const std::pair<int, int>& _end) const
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

	LinAlg::Matrix sub_matrix({ rows, columns }, 0.0);

	for (int row = _start.first, r = 0; row < _end.first; row++, r++)
	{
		for (int col = _start.second, c = 0; col < _end.second; col++, c++)
		{
			sub_matrix.data[r][c] = this->data[row][col];
		}
	}

	return sub_matrix;
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

	LinAlg::Matrix sub_matrix;

	for (int row = 0; row < this->shape.first; row++)
	{
		if (row_filter[row])
		{
			sub_matrix.PushRow(this->GetRow(row));
		}
	}

	return sub_matrix;
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

	LinAlg::Matrix sub_matrix;

	for (int i = 0; i < this->shape.second; i++)
	{
		if (column_filter[i])
		{
			sub_matrix.PushColumn(this->GetColumn(i));
		}
	}

	return sub_matrix;
}

std::vector<double> LinAlg::Matrix::GetFlatData() const
{
	std::vector<double> flat_data;

	if (this->IsEmpty())
	{
		return flat_data;
	}

	flat_data.reserve(this->volume);

	for (int i = 0; i < this->shape.first; i++)
	{
		flat_data.insert(flat_data.end(), this->data[i].begin(), this->data[i].end());
	}

	return flat_data;
}

// ========================================
// Matrix Row/Column Append Method(s)
// ========================================
void LinAlg::Matrix::PushRow(const std::vector<double>& _row_data)
{
	if (!this->IsEmpty() && _row_data.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Row Appending failed: row array-size mismatch with Matrix column-size.");
	}

	if (!Utils::IsValidData(_row_data))
	{
		throw std::invalid_argument("[Matrix] Row Appending failed: invalid value found in row-data.");
	}

	this->shape.first++;
	this->volume += _row_data.size();
	this->data.push_back(_row_data);
}

void LinAlg::Matrix::PushColumn(const std::vector<double>& _column_data)
{
	if (!this->IsEmpty() && _column_data.size() != this->shape.second)
	{
		throw std::invalid_argument("[Matrix] Row Appending failed: row array-size mismatch with Matrix column-size.");
	}

	if (!Utils::IsValidData(_column_data))
	{
		throw std::invalid_argument("[Matrix] Row Appending failed: invalid value found in row-data.");
	}

	this->shape.second++;
	this->volume += _column_data.size();

	if (this->data.empty())
	{
		this->data.resize(_column_data.size());
		for (const double& value : _column_data)
		{
			this->data.push_back({ value });
		}
	}
	else
	{
		for (int row = 0; row < _column_data.size(); row++)
		{
			this->data[row].push_back(_column_data[row]);
		}
	}
}

// ========================================
// Matrix Row(s)/Column(s) Removal Method(s)
// ========================================
void LinAlg::Matrix::PopRow(const int& _index)
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Pop Row failed: empty Matrix.");
	}

	int index = (_index < 0) ? (_index + static_cast<int>(this->shape.first)) : _index;

	if (index < 0 || static_cast<size_t>(index) >= this->shape.first)
	{
		throw std::out_of_range("[Matrix] Pop Row failed: index: " + std::to_string(index) + " out of bounds: [0, rows).");
	}

	this->data.erase(this->data.begin() + index);

	if (this->data.empty() || this->data[0].empty())
	{
		this->data.clear();
		this->shape = { 0, 0 };
		this->volume = 0;
	}
	else
	{
		this->shape = { this->data.size(), this->data[0].size() };
		this->volume = this->shape.first * this->shape.second;
	}
}

void LinAlg::Matrix::PopColumn(const int& _index)
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Pop Row failed: empty Matrix.");
	}

	int index = (_index < 0) ? (_index + this->shape.second) : _index;

	if (index < 0 || index >= this->shape.second)
	{
		throw std::out_of_range("[Matrix] Pop Column failed: index: " + std::to_string(index) + " out of bounds: [0, columns).");
	}

	for (int i = 0; i < this->shape.first; i++)
	{
		this->data[i].erase(this->data[i].begin() + index);
	}

	if (this->data.size() == 0 || this->data[0].size() == 0)
	{
		this->data.clear();
		this->shape = { 0, 0 };
		this->volume = 0;
	}
	else
	{
		this->shape = { this->data.size(), this->data[0].size() };
		this->volume = this->shape.first * this->shape.second;
	}
}

void LinAlg::Matrix::PopRows(const std::vector<int>& _indices)
{
	if (!Utils::IsBounded(_indices, this->shape.first, -1, true))
	{
		throw std::out_of_range("[Matrix] Pop Rows failed: index(s) out of bounds: [0, rows).");
	}

	if (!Utils::IsAllUnique(_indices))
	{
		throw std::invalid_argument("[Matrix] Pop Rows failed: index values must be unique.");
	}

	std::vector<int> desc_indices = _indices;
	std::sort(desc_indices.begin(), desc_indices.end(), std::greater<int>());

	for (const int& index : desc_indices)
	{
		this->PopRow(index);
	}
}

void LinAlg::Matrix::PopColumns(const std::vector<int>& _indices)
{
	if (!Utils::IsBounded(_indices, this->shape.second, -1, true))
	{
		throw std::out_of_range("[Matrix] Pop Columns failed: index(s) out of bounds: [0, columns).");
	}

	if (!Utils::IsAllUnique(_indices))
	{
		throw std::invalid_argument("[Matrix] Pop Columns failed: index values must be unique.");
	}

	std::vector<int> desc_indices = _indices;
	std::sort(desc_indices.begin(), desc_indices.end(), std::greater<int>());

	for (const int& index : desc_indices)
	{
		this->PopColumn(index);
	}
}

// ========================================
// Matrix Norm Computation Method(s)
// ========================================
double LinAlg::Matrix::FrobeniusNorm() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Frobenius-Norm Computation failed: empty Matrix.");
	}

	double frobenius_norm = 0.0;

	for (int row = 0; row < this->shape.first; row++)
	{
		for (int col = 0; col < this->shape.second; col++)
		{
			frobenius_norm += (this->data[row][col] * this->data[row][col]);
		}
	}

	frobenius_norm = std::sqrt(frobenius_norm);

	return frobenius_norm;
}

double LinAlg::Matrix::SpectralNorm() const
{
	// Build after SVD is done.
	return 0.0;
}

double LinAlg::Matrix::NuclearNorm() const
{
	// Build after SVD is done.
	return 0.0;
}

double LinAlg::Matrix::InfinityNorm() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Infinity-Norm Computation failed: empty Matrix.");
	}

	double infinity_norm = 0.0;

	for (int row = 0; row < this->shape.first; row++)
	{
		double sum = 0.0;
		for (int col = 0; col < this->shape.second; col++)
		{
			sum += std::abs(this->data[row][col]);
		}

		infinity_norm = std::max(sum, infinity_norm);
	}

	return infinity_norm;
}

double LinAlg::Matrix::OneNorm() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] One-Norm Computation failed: empty Matrix.");
	}

	double one_norm = 0.0;

	for (int col = 0; col < this->shape.second; col++)
	{
		double sum = 0.0;
		for (int row = 0; row < this->shape.first; row++)
		{
			sum += std::abs(this->data[row][col]);
		}

		one_norm = std::max(sum, one_norm);
	}

	return one_norm;
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

	return LinAlg::LUResult(L, U, P);
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

LinAlg::QRResult LinAlg::Matrix::GSQRDecomposition() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] QR Decomposition failed: empty matrix.");
	}

	int rows = this->shape.first;
	int columns = this->shape.second;

	if (rows < columns)
	{
		throw std::runtime_error(
			"[Matrix] Gram-Schmidt QR Decomposition failed: only supports m ? n (tall or square matrices). "
			"For wide matrices (m < n), use HTQRDecomposition() instead. "
			"Current shape: (" + std::to_string(rows) + ", " + std::to_string(columns) + ")"
		);
	}

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
			throw std::runtime_error(
				"[Matrix] Gram-Schmidt QR Decomposition failed: linearly dependent columns at column " +
				std::to_string(i)
			);
		}

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

LinAlg::QRResult LinAlg::Matrix::HQRDecomposition(const bool& _full) const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Householder QR Decomposition failed: empty matrix.");
	}

	int rows = this->shape.first;
	int columns = this->shape.second;

	int k = std::min(rows, columns);

	LinAlg::Matrix Q = LinAlg::Matrix::Identity(rows);
	LinAlg::Matrix R = *this;

	for (int i = 0; i < k; i++)
	{
		int vsize = rows - i;
		std::vector<double> x(vsize);

		for (int j = i; j < rows; j++)
		{
			x[j - i] = R.data[j][i];
		}

		double norm_x = Utils::Norm(x, 2);
		if (std::abs(norm_x) < LinAlg::Matrix::TOLERANCE)
		{
			continue;
		}

		x[0] += (x[0] >= 0) ? norm_x : -norm_x;

		norm_x = Utils::Norm(x, 2);
		if (std::abs(norm_x) < LinAlg::Matrix::TOLERANCE)
		{
			continue;
		}

		for (double& value : x)
		{
			value /= norm_x;
		}

		LinAlg::Matrix v_col({ vsize, 1 }, x);
		LinAlg::Matrix v_row = v_col.Transpose();
		LinAlg::Matrix vvt = v_col.MatMul(v_row);
		LinAlg::Matrix h_sub = LinAlg::Matrix::Identity(vsize) - (vvt * 2);

		LinAlg::Matrix Qi = LinAlg::Matrix::Identity(rows);

		for (int r = 0; r < vsize; r++)
		{
			for (int c = 0; c < vsize; c++)
			{
				Qi.data[r + i][c + i] = h_sub.data[r][c];
			}
		}

		Q = Q.MatMul(Qi);
		R = Qi.MatMul(R);
	}

	if (!_full && rows > columns)
	{
		int diff = rows - k;
		
		std::vector<int> indices(diff);
		std::iota(indices.begin(), indices.end(), k);

		Q.PopColumns(indices);
		R.PopRows(indices);
	}

	Q.ClearNoise();
	R.ClearNoise();

	return LinAlg::QRResult(Q, R);
}

LinAlg::SVDResult LinAlg::Matrix::SVDecomposition() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] SVDecomposition failed: empty Matrix.");
	}

	LinAlg::GKBResult ubv_t = this->GKBidiagonalize(); // this part is working fine.
	LinAlg::SVDResult u1sv1_t = ubv_t.B.GRDiagonalize();

	LinAlg::Matrix U = LinAlg::Matrix::MatMul(ubv_t.U, u1sv1_t.U);
	LinAlg::Matrix V = LinAlg::Matrix::MatMul(ubv_t.V, u1sv1_t.V);

	return LinAlg::SVDResult(U, u1sv1_t.S, V);
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

LinAlg::GKBResult LinAlg::Matrix::GKBidiagonalize() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Golub-Kahan-Bidiagonalization failed: empty matrix.");
	}

	int rows = this->shape.first;
	int columns = this->shape.second;

	LinAlg::Matrix U = LinAlg::Matrix::Identity(rows);
	LinAlg::Matrix B = *this;
	LinAlg::Matrix V = LinAlg::Matrix::Identity(columns);

	for (int i = 0; i < columns; i++)
	{
		if (i < rows)
		{
			int vsize = rows - i;
			std::vector<double> x(vsize);

			for (int j = i; j < rows; j++)
			{
				x[j - i] = B.data[j][i];
			}

			double norm_x = Utils::Norm(x, 2);
			if (std::abs(norm_x) < LinAlg::Matrix::TOLERANCE)
			{
				continue;
			}

			x[0] += (x[0] >= 0) ? norm_x : -norm_x;

			norm_x = Utils::Norm(x, 2);
			if (std::abs(norm_x) < LinAlg::Matrix::TOLERANCE)
			{
				continue;
			}

			for (double& value : x)
			{
				value /= norm_x;
			}

			LinAlg::Matrix v_col({ vsize, 1 }, x);
			LinAlg::Matrix v_row = v_col.Transpose();
			LinAlg::Matrix vvt = v_col.MatMul(v_row);
			LinAlg::Matrix h_sub = LinAlg::Matrix::Identity(vsize) - (vvt * 2);

			LinAlg::Matrix Ui = LinAlg::Matrix::Identity(rows);

			for (int r = 0; r < vsize; r++)
			{
				for (int c = 0; c < vsize; c++)
				{
					Ui.data[r + i][c + i] = h_sub.data[r][c];
				}
			}

			U = U.MatMul(Ui);
			B = Ui.MatMul(B);
		}

		if (i < (columns - 1))
		{
			int vsize = columns - i - 1;
			std::vector<double> x(vsize);

			for (int j = i + 1; j < columns; j++)
			{
				x[j - i - 1] = B.data[i][j];
			}

			double norm_x = Utils::Norm(x, 2);
			if (std::abs(norm_x) < LinAlg::Matrix::TOLERANCE)
			{
				continue;
			}

			x[0] += (x[0] >= 0) ? norm_x : -norm_x;
			norm_x = Utils::Norm(x, 2);

			if (std::abs(norm_x) < LinAlg::Matrix::TOLERANCE)
			{
				continue;
			}

			for (double& value : x)
			{
				value /= norm_x;
			}

			LinAlg::Matrix v_col({ vsize, 1 }, x);
			LinAlg::Matrix v_row = v_col.Transpose();
			LinAlg::Matrix vvt = v_col.MatMul(v_row);
			LinAlg::Matrix h_sub = LinAlg::Matrix::Identity(vsize) - (vvt * 2);

			LinAlg::Matrix Vi = LinAlg::Matrix::Identity(columns);

			for (int r = 0; r < vsize; r++)
			{
				for (int c = 0; c < vsize; c++)
				{
					Vi.data[r + i + 1][c + i + 1] = h_sub.data[r][c];
				}
			}

			B = B.MatMul(Vi);
			V = V.MatMul(Vi);
		}
	}

	return LinAlg::GKBResult(U, B, V);
}

LinAlg::SVDResult LinAlg::Matrix::GRDiagonalize() const
{
	if (this->IsEmpty())
	{
		throw std::runtime_error("[Matrix] Golub-Reinsch-Diagonalization failed: empty Matrix.");
	}

	if (!this->IsBidiagonal())
	{
		throw std::runtime_error("[Matrix] Golub-Reinsch-Diagonalization failed: requires a bidiagonal Matrix.");
	}

	int rows = this->shape.first;
	int columns = this->shape.second;
	int k = std::min(rows, columns);

	LinAlg::Matrix S = *this;
	LinAlg::Matrix U = LinAlg::Matrix::Identity(rows);
	LinAlg::Matrix V = LinAlg::Matrix::Identity(columns);

	const int max_iteration = (100 * columns);

	for (int itr = 0; itr < max_iteration; itr++)
	{
		double max_offdiag = 0.0;

		for (int i = 0; i < k - 1; i++)
		{
			max_offdiag = std::max(max_offdiag, std::abs(S.data[i][i + 1]));
		}

		if (max_offdiag < LinAlg::Matrix::TOLERANCE)
		{
			break;
		}

		for (int i = 0; i < k - 1; i++)
		{
			double diag_i = std::abs(S.data[i][i]);
			double diag_i1 = (i + 1 < k) ? std::abs(S.data[i + 1][i + 1]) : 0.0;

			if (std::abs(S.data[i][i + 1]) < LinAlg::Matrix::TOLERANCE * (diag_i + diag_i1))
			{
				S.data[i][i + 1] = 0.0;
			}
		}

		int q = k - 1;
		while (q > 0 && std::abs(S.data[q - 1][q]) < LinAlg::Matrix::TOLERANCE)
		{
			q--;
		}

		if (q == 0) break;

		int p = q - 1;
		while (p > 0 && std::abs(S.data[p - 1][p]) >= LinAlg::Matrix::TOLERANCE)
		{
			p--;
		}

		double mu = 0.0;
		{
			double d1 = S.data[q - 1][q - 1];
			double d2 = S.data[q][q];
			double e1 = (q > 1) ? S.data[q - 2][q - 1] : 0.0;
			double e2 = S.data[q - 1][q];

			double a11 = d1 * d1 + e1 * e1;
			double a22 = d2 * d2 + e2 * e2;
			double a12 = d1 * e2;

			LinAlg::Matrix T({ 2, 2 }, { a11, a12, a12, a22 });
			mu = T.WilkinsonShift();
		}

		double d_p = S.data[p][p];
		double e_p = (p < k - 1) ? S.data[p][p + 1] : 0.0;

		double y = (d_p * d_p) - mu;
		double z = (d_p * e_p);

		for (int i = p; i < q; i++)
		{
			// Right Givens
			auto [c1, s1] = LinAlg::Matrix::ComputeGivens(y, z);
			LinAlg::Matrix G({2, 2}, { c1, s1, -s1, c1 });

			S = S.PartialMatMul(G, { i, i }, { i + 2, i + 2 }, false);
			V = V.PartialMatMul(G, { i, i }, { i + 2, i + 2 }, false);

			// Left Givens
			y = S.data[i][i];
			z = ((i + 1) < rows) ? S.data[i + 1][i] : 0.0;

			auto [c2, s2] = LinAlg::Matrix::ComputeGivens(y, z);
			LinAlg::Matrix H({ 2, 2 }, { c2, -s2, s2, c2 });

			S = S.PartialMatMul(H, { i, i }, { i + 2, i + 2 }, true);
			U = U.PartialMatMul(H.Transpose(), {i, i}, {i + 2, i + 2}, false);

			// Update y, z for next iteration
			if (i < q - 1)
			{
				y = S.data[i][i + 1];
				z = ((i + 2) < columns) ? S.data[i][i + 2] : 0.0;
			}
		}
	}

	U.ClearNoise();
	S.ClearNoise();
	V.ClearNoise();
	

	std::vector<std::pair<double, int>> singular_values(k);

	for (int i = 0; i < k; i++)
	{
		singular_values[i] = std::make_pair(S.data[i][i], i);
	}

	std::sort(singular_values.begin(), singular_values.end(),
		[](const std::pair<double, int>& a, const std::pair<double, int>& b)
		{
			return std::abs(a.first) > std::abs(b.first);
		});

	std::vector<int> left_permutation(rows);
	std::vector<int> right_permutation(columns);

	for (int i = 0; i < rows; i++)
	{
		left_permutation[i] = (i < k) ? singular_values[i].second : i;
	}

	for (int i = 0; i < columns; i++)
	{
		right_permutation[i] = (i < k) ? singular_values[i].second : i;
	}

	S.PermuteRows(left_permutation);
	S.PermuteColumns(right_permutation);

	U.PermuteColumns(left_permutation);
	V.PermuteColumns(right_permutation);

	for (int i = 0; i < k; i++)
	{
		if (S.data[i][i] < 0.0)
		{
			S.data[i][i] = -S.data[i][i];
			for (int row = 0; row < rows; row++)
			{
				U.data[row][i] = -U.data[row][i];
			}
		}
	}

	return LinAlg::SVDResult(U, S, V);
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
			std::cout << value << "\t";
		}
		std::cout << std::endl;
	}
}
