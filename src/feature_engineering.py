# 本文件存放用于手工摘取特征的函数。

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # * 年份可以用差值来代替建造时间
        X_input = X.copy()
        X_input["GarageAge"] = 0
        mask = X_input["HasGarage"] == 1
        X_input.loc[mask, "GarageAge"] = (X_input.loc[mask, "YrSold"] - X_input.loc[mask, "GarageYrBlt"])
        X_input["HouseAge"] = X_input["YrSold"] - X_input["YearBuilt"]
        X_input["RemodAge"] = X_input["YrSold"] - X_input["YearRemodAdd"]

        # * 可以把所有的面积加起来计算总使用面积
        X_input["TotalUtilSF"] = X_input["1stFlrSF"] + X_input["2ndFlrSF"] + X_input["TotalBsmtSF"]

        # * 卫生间等效加总
        X_input["EqvlBath"] = X_input["BsmtFullBath"] + 0.5 * X_input["BsmtHalfBath"] + X_input["FullBath"] + 0.5 * X_input["HalfBath"]

        # * 比例系列
        X_input["LivingRatio"] = X_input["GrLivArea"] / X_input["LotArea"]
        X_input["BedroomRatio"] = X_input["Bedroom"] / X_input["TotalUtilSF"]

        # * 面积×质量
        X_input["OvrallMul"] = X_input["OverallQual"] * X_input["LotArea"]
        X_input["BsmtMul"] = X_input["BsmtQual"] * X_input["TotalBsmtSF"]
        X_input["KitMul"] = X_input["Kitchen"] * X_input["KitchenQual"] # 这个是数量 * 质量

        # * Porch 总面积
        X_input["TotalPorchSF"] = X_input["OpenPorchSF"] + X_input["EnclosedPorch"] + X_input["3SsnPorch"] + X_input["ScreenPorch"]

        # * 设施存在性
        # HasGarage已经有了
        X_input["HasPool"] = 1 * (X_input["PoolArea"] > 0)
        X_input["HasBsmt"] = 1 * (X_input["BsmtQual"] > 0)
        X_input["HasFireplace"] = 1 * (X_input["Fireplaces"] > 0)
        X_input["Has2ndfloor"] = 1 * (X_input["2ndFlrSF"] > 0)
        X_input["HasPorch"] = 1 * (X_input["TotalPorchSF"] > 0)
        X_input["HasFence"] = 1 * (X_input["Fence"] > 0)

        # * 总设施数量
        X_input["Totalutil"] = X_input["HasPool"] + X_input["HasGarage"] + X_input["HasBsmt"] + X_input["HasFireplace"] + X_input["Has2ndfloor"] + X_input["HasPorch"] + X_input["HasFence"]

        # * 根号项
        X_input["RootUtil"] = X_input["Totalutil"] ** 0.5
        X_input["RootLotArea"] = X_input["LotArea"] ** 0.5

        return X_input






