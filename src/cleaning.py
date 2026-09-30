# 本脚本用于存放数据清理的工具函数
import pandas as pd


def clean_data(df_train, test_data = False):
    # ================================================
    # 错误列名处理
    # ================================================
    df_train.rename(columns={
        "BedroomAbvGr": "Bedroom",
        "KitchenAbvGr": "Kitchen"
    }, inplace=True)

    print("错误列名矫正完成.")

    # ================================================
    # 统计
    # ================================================
    print("存在缺失值NA的列:")
    series = df_train.isna().mean()
    series = series[series != 0]
    print(series.sort_values().to_string())

    # ================================================
    # 处理缺失值
    # ================================================
    # 处理MiscFeature
    # 取出对应列，对该列赋值, 使用fillna()
    df_train["MiscFeature"] = df_train["MiscFeature"].fillna("NoMisc")

    # 处理Alley
    df_train["Alley"] = df_train["Alley"].fillna("NoAlley")

    # PoolQC把NA填成NoPool
    df_train["PoolQC"] = df_train["PoolQC"].fillna("NoPool")

    # FireplaceQu把NA填成NoFireplace
    df_train["FireplaceQu"] = df_train["FireplaceQu"].fillna("NoFireplace")

    # 处理Garage系列: GarageType, GarageQual, GarageYrBlt, GarageFinish, GarageCond
    df_train["GarageType"] = df_train["GarageType"].fillna("NoGarage")
    df_train["GarageQual"] = df_train["GarageQual"].fillna("NoGarage")
    df_train["GarageYrBlt"] = df_train["GarageYrBlt"].fillna("NoGarage")
    df_train["GarageFinish"] = df_train["GarageFinish"].fillna("NoGarage")
    df_train["GarageCond"] = df_train["GarageCond"].fillna("NoGarage")

    # 处理Bsmt系列：BsmtFinType1, BsmtQual, BsmtCond和 BsmtFinType2, BsmtExposure
    df_train["BsmtFinType1"] = df_train["BsmtFinType1"].fillna("NoBsmt")
    df_train["BsmtQual"] = df_train["BsmtQual"].fillna("NoBsmt")
    df_train["BsmtCond"] = df_train["BsmtCond"].fillna("NoBsmt")
    df_train["BsmtFinType2"] = df_train["BsmtFinType2"].fillna("NoBsmt")
    df_train["BsmtExposure"] = df_train["BsmtExposure"].fillna("NoBsmt")

    # 这里做一下判断，如果不是测试数据，就可以删列，如果是，就换成mode
    if not test_data:
        # 删去Electrical 列中，为空的行
        df_train = df_train[~df_train["Electrical"].isna()]
    else:
        df_train["Electrical"] = df_train["Electrical"].fillna(df_train["Electrical"].mode()[0])


    # 取出LotFrontage列的中位数
    LotFrontage_median = df_train["LotFrontage"].median()
    # 然后把所有为NaN的替换成中位数
    df_train["LotFrontage"] = df_train["LotFrontage"].fillna(LotFrontage_median)

    # Fence，NA fillna为NoFence
    df_train["Fence"] = df_train["Fence"].fillna("NoFence")

    # 把所有pandas认为是空缺的填补上NoMasVnr
    df_train["MasVnrType"] = df_train["MasVnrType"].fillna("NoMasVnr")

    # MasVnrArea直接删除空缺行
    # 改成只有训练数据删行
    if not test_data:
        df_train = df_train[~df_train["MasVnrArea"].isna()]
    else:
        df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)

    # Garage系列空缺处理
    df_train["HasGarage"] = 1 * (df_train["GarageYrBlt"] != "NoGarage")
    df_train["GarageYrBlt"] = df_train["GarageYrBlt"].replace("NoGarage", 0)
    df_train["GarageYrBlt"] = df_train["GarageYrBlt"].astype(int)

    # 汇报
    print("缺失值清理完成.")
    # 再次统计
    print("清理一次之后，存在缺失值NA的列:")
    series = df_train.isna().mean()
    series = series[series != 0]
    print(series.sort_values().to_string())
    # ================================================
    # 处理额外缺失值
    # ================================================
    if test_data:
        df_train["Exterior2nd"] = df_train["Exterior2nd"].fillna(df_train["Exterior2nd"].mode()[0])
        df_train["Exterior1st"] = df_train["Exterior1st"].fillna(df_train["Exterior1st"].mode()[0])
        df_train["BsmtFinSF2"] = df_train["BsmtFinSF2"].fillna(df_train["BsmtFinSF2"].mean())
        df_train["BsmtFinSF1"] = df_train["BsmtFinSF1"].fillna(df_train["BsmtFinSF1"].mean())
        df_train["BsmtUnfSF"] = df_train["BsmtUnfSF"].fillna(df_train["BsmtUnfSF"].mean())
        df_train["TotalBsmtSF"] = df_train["TotalBsmtSF"].fillna(df_train["TotalBsmtSF"].mean())
        df_train["SaleType"] = df_train["SaleType"].fillna(df_train["SaleType"].mode()[0])
        df_train["KitchenQual"] = df_train["KitchenQual"].fillna(df_train["KitchenQual"].mode()[0])
        df_train["GarageArea"] = df_train["GarageArea"].fillna(df_train["GarageArea"].mean())
        df_train["GarageCars"] = df_train["GarageCars"].fillna(df_train["GarageCars"].mode()[0])
        df_train["Functional"] = df_train["Functional"].fillna(df_train["Functional"].mode()[0])
        df_train["Utilities"] = df_train["Utilities"].fillna(df_train["Utilities"].mode()[0])
        df_train["BsmtFullBath"] = df_train["BsmtFullBath"].fillna(df_train["BsmtFullBath"].mode()[0])
        df_train["BsmtHalfBath"] = df_train["BsmtHalfBath"].fillna(df_train["BsmtHalfBath"].mode()[0])
        df_train["MSZoning"] = df_train["MSZoning"].fillna(df_train["MSZoning"].mode()[0])

        # 再次统计
        print("清理两次之后，存在缺失值NA的列:")
        series = df_train.isna().mean()
        series = series[series != 0]
        print(series.sort_values().to_string())
    # ================================================
    # 转换类别变量
    # ================================================
    # 删除Id，保存
    # df_train["Id"].to_csv(PROCESSED_DIR / "Id.csv", index=False)
    df_train.drop("Id", axis=1, inplace=True)

    # 替换MSSubClass的类型
    df_train["MSSubClass"] = df_train["MSSubClass"].astype("str")

    # * LotShape
    df_train["LotShape"] = df_train["LotShape"].replace(
        {
            "Reg": 1,
            "IR1": 2,
            "IR2": 3,
            "IR3": 4
        }
    )
    df_train["LotShape"] = df_train["LotShape"].astype(int)

    # * Utilities
    df_train["Utilities"] = df_train["Utilities"].replace(
        {
            "AllPub": 4,
            "NoSewr": 3,
            "NoSeWa": 2,
            "ELO": 1
        }
    )
    df_train["Utilities"] = df_train["Utilities"].astype(int)

    # * LandSlope
    df_train["LandSlope"] = df_train["LandSlope"].replace(
        {
            "Gtl": 1,
            "Mod": 2,
            "Sev": 3
        }
    )
    df_train["LandSlope"] = df_train["LandSlope"].astype(int)

    # * ExterQual
    df_train["ExterQual"] = df_train["ExterQual"].replace(
        {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1
        }
    )
    df_train["ExterQual"] = df_train["ExterQual"].astype(int)

    # * ExterCond
    df_train["ExterCond"] = df_train["ExterCond"].replace(
        {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1
        }
    )
    df_train["ExterCond"] = df_train["ExterCond"].astype(int)

    # * BsmtQual
    df_train["BsmtQual"] = df_train["BsmtQual"].replace(
        {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NoBsmt": 0
        }
    )
    df_train["BsmtQual"] = df_train["BsmtQual"].astype(int)

    # * BsmtCond
    df_train["BsmtCond"] = df_train["BsmtCond"].replace(
        {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NoBsmt": 0
        }
    )
    df_train["BsmtCond"] = df_train["BsmtCond"].astype(int)

    # * BsmtExposure
    df_train["BsmtExposure"] = df_train["BsmtExposure"].replace(
        {
            "Gd": 4,
            "Av": 3,
            "Mn": 2,
            "No": 1,
            "NoBsmt": 0
        }
    )
    df_train["BsmtExposure"] = df_train["BsmtExposure"].astype(int)

    # * BsmtFinType1
    df_train["BsmtFinType1"] = df_train["BsmtFinType1"].replace(
        {
            "GLQ": 6,
            "ALQ": 5,
            "BLQ": 4,
            "Rec": 3,
            "LwQ": 2,
            "Unf": 1,
            "NoBsmt": 0
        }
    )
    df_train["BsmtFinType1"] = df_train["BsmtFinType1"].astype(int)

    # * BsmtFinType2
    df_train["BsmtFinType2"] = df_train["BsmtFinType2"].replace(
        {
            "GLQ": 6,
            "ALQ": 5,
            "BLQ": 4,
            "Rec": 3,
            "LwQ": 2,
            "Unf": 1,
            "NoBsmt": 0
        }
    )
    df_train["BsmtFinType2"] = df_train["BsmtFinType2"].astype(int)

    # * HeatingQC
    df_train["HeatingQC"] = df_train["HeatingQC"].replace(
        {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
        }
    )
    df_train["HeatingQC"] = df_train["HeatingQC"].astype(int)

    # * KitchenQual
    df_train["KitchenQual"] = df_train["KitchenQual"].replace(
        {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
        }
    )
    df_train["KitchenQual"] = df_train["KitchenQual"].astype(int)

    # * Functional
    df_train["Functional"] = df_train["Functional"].replace(
        {
            "Typ": 7,
            "Min1": 6,
            "Min2": 5,
            "Mod": 4,
            "Maj1": 3,
            "Maj2": 2,
            "Sev": 1,
            "Sal": 0
        }
    )
    df_train["Functional"] = df_train["Functional"].astype(int)

    # * FireplaceQu
    df_train["FireplaceQu"] = df_train["FireplaceQu"].replace(
        {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NoFireplace": 0
        }
    )
    df_train["FireplaceQu"] = df_train["FireplaceQu"].astype(int)

    # * GarageFinish
    df_train["GarageFinish"] = df_train["GarageFinish"].replace(
        {
            "Fin": 3,
            "RFn": 2,
            "Unf": 1,
            "NoGarage": 0
        }
    )
    df_train["GarageFinish"] = df_train["GarageFinish"].astype(int)

    # * GarageQual
    df_train["GarageQual"] = df_train["GarageQual"].replace(
        {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NoGarage": 0
        }
    )
    df_train["GarageQual"] = df_train["GarageQual"].astype(int)

    # * GarageCond
    df_train["GarageCond"] = df_train["GarageCond"].replace(
        {
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NoGarage": 0
        }
    )
    df_train["GarageCond"] = df_train["GarageCond"].astype(int)

    # * PavedDrive
    df_train["PavedDrive"] = df_train["PavedDrive"].replace(
        {
            "Y": 2,
            "P": 1,
            "N": 0
        }
    )
    df_train["PavedDrive"] = df_train["PavedDrive"].astype(int)

    # * PoolQC
    df_train["PoolQC"] = df_train["PoolQC"].replace(
        {
            "Ex": 4,
            "Gd": 3,
            "TA": 2,
            "Fa": 1,
            "NoPool": 0
        }
    )
    df_train["PoolQC"] = df_train["PoolQC"].astype(int)
    # * Fence
    df_train["Fence"] = df_train["Fence"].replace(
        {
            "GdPrv": 4,
            "MnPrv": 3,
            "GdWo": 2,
            "MnWw": 1,
            "NoFence": 0
        }
    )
    df_train["Fence"] = df_train["Fence"].astype(int)


    return df_train

