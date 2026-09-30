# 本数据接受已经清理过的数据集，然后实施预处理
# 包括清理过的数据集，或者经过特征工程，或者特征选择后的数据集
# 数据集类型应当仅包括数值和字符串变量

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import RobustScaler, OneHotEncoder


def build_preprocessor():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num", RobustScaler(),
                make_column_selector(dtype_include="number")
            ),

            (
                "cat", OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False
                ),
                make_column_selector(dtype_exclude="number")
            )
        ]
    )

    return preprocessor

