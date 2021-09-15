import pandas as pd
from functools import reduce

import datetime as dt


def make_date_series_slice(df, day):
    start = dt.date(df.year.unique()[0], 4, 3)
    end = dt.date(df.year.unique()[0], 9, 8)
    print(f'{df.NAME.unique()[0]}')
    df = df[(df.Date.dt.day >= start.day) & (df.Date.dt.month >= start.month) &
            (df.Date.dt.day <= end.day) & (df.Date.dt.month <= end.month)]

    if df["Date"].iloc[0] != start:
        df.loc[-1] = 0
        df.loc[-1, 'Date'] = pd.to_datetime(start)
        df.index = df.index + 1
        df.sort_index(inplace=True)

    if df["Date"].iloc[-1] != end:
        df.loc[len(df) + 1] = 0
        df.loc[len(df), 'Date'] = pd.to_datetime(end)

    df_resample = df.set_index('Date').resample(f'{day}D').sum().reset_index()

    X_demo = ['PCODE', 'NAME']

    X_feature = ['선발', '타수', '득점', '안타', '2타', '3타', '홈런',
                 '루타', '타점', '도루', '도실', '볼넷', '사구', '고4', '삼진',
                 '병살', '희타', '희비', '투구', 'barrel', '장타', '출루']

    on_base = ['안타', '2타', '3타', '홈런', '사구', '볼넷', '고4']
    total_base = ['타수', '볼넷', '고4', '사구', '희타']
    hit = ['안타', '2타', '3타', '홈런']

    df_resample['출루'] = df_resample[on_base].sum(axis=1) / df_resample[total_base].sum(axis=1)
    df_resample['장타'] = (df_resample[hit] * [1, 2, 3, 4]).sum(axis=1) / df_resample['타수']

    df_resample['PCODE'] = [df['PCODE'].unique().astype(int)[0] for _ in range(df_resample.shape[0])]
    df_resample['NAME'] = [df['NAME'].unique().astype(object)[0] for _ in range(df_resample.shape[0])]
    df_resample.index = [f't{i}' for i in range(df_resample.shape[0])][::-1]

    df_resample[['장타', '출루']] = df_resample[['장타', '출루']].fillna(0)

    df_resample['장타'] = list(df_resample['장타'].iloc[1:]) + [None]
    df_resample['출루'] = list(df_resample['출루'].iloc[1:]) + [None]

    return df_resample[X_demo + X_feature]


def make_date_series(df, day):


    start = dt.date(df.year.unique()[0], 4, 3)
    end = dt.date(df.year.unique()[0], 9, 8)
    print(f'{df.NAME.unique()[0]}')
    df = df[(df.Date.dt.day >= start.day) & (df.Date.dt.month >= start.month) &
            (df.Date.dt.day <= end.day) & (df.Date.dt.month <= end.month)]

    if df["Date"].iloc[0] != start:
        df.loc[-1] = 0
        df.loc[-1, 'Date'] = pd.to_datetime(start)
        df.index = df.index + 1
        df.sort_index(inplace=True)

    if df["Date"].iloc[-1] != end:
        df.loc[len(df) + 1] = 0
        df.loc[len(df), 'Date'] = pd.to_datetime(end)

    df_resample = df.set_index('Date').resample(f'{day}D').sum().reset_index()

    X_demo = ['PCODE', 'NAME']

    X_feature = ['선발', '타수', '득점', '안타', '2타', '3타', '홈런',
                 '루타', '타점', '도루', '도실', '볼넷', '사구', '고4', '삼진',
                 '병살', '희타', '희비', '투구', 'barrel', '장타', '출루']

    on_base = ['안타', '2타', '3타', '홈런', '사구', '볼넷', '고4']
    total_base = ['타수', '볼넷', '고4', '사구', '희타']
    hit = ['안타', '2타', '3타', '홈런']

    df_resample['출루'] = df_resample[on_base].sum(axis=1) / df_resample[total_base].sum(axis=1)
    df_resample['장타'] = (df_resample[hit] * [1, 2, 3, 4]).sum(axis=1) / df_resample['타수']

    df_resample['PCODE'] = [df['PCODE'].unique().astype(int)[0] for _ in range(df_resample.shape[0])]
    df_resample['NAME'] = [df['NAME'].unique().astype(object)[0] for _ in range(df_resample.shape[0])]
    df_resample.index = [f't{i}' for i in range(df_resample.shape[0])][::-1]

    df_resample[['장타', '출루']] = df_resample[['장타', '출루']].fillna(0)



    return df_resample[X_demo + X_feature]

def make_time_series_barrel(df, window_size):

    X_demo = ['PCODE']

    X_feature = ['선발', '타수', '득점', '안타', '2타', '3타', '홈런',
                 '루타', '타점', '도루', '도실', '볼넷', '사구', '고4', '삼진',
                 '병살', '희타', '희비', '투구', 'barrel']


    on_base = ['안타', '2타', '3타', '홈런', '사구', '볼넷', '고4']
    total_base = ['타수', '볼넷', '고4', '사구', '희타']
    hit = ['안타', '2타', '3타', '홈런']

    X_out= ['장타', '출루']


    sample_list = []

    for i in range(1, len(df), window_size):
        single = df[-i:-(i + window_size):-1].groupby(['PCODE', 'year'])[X_feature].sum()
        single['gap'] = df[-i:-(i + window_size):-1].iloc[0, -3] - df[-i:-(i + window_size):-1].iloc[-1, -3]

        single['num_game'] = df[-i:-(i + window_size):-1].shape[0]
        single['출루'] = single[on_base].sum(axis=1) / single[total_base].sum(axis=1)
        single['장타'] = (single[hit] * [1, 2, 3, 4]).sum(axis=1) / single['타수']
        sample_list.append(single)

    result = pd.concat(sample_list).reset_index()
    result.index = ['t' + str(i) for i in range(1, len(result) + 1)]

    result['장타'] = [None] + list(result['장타'].iloc[:-1])
    result['출루'] = [None] + list(result['출루'].iloc[:-1])

    return result[X_demo+X_feature+X_out]


def make_time_series(df, window_size, slice):


    if slice:
        X_demo = ['PCODE', 'NAME']

        X_feature = ['선발', '타수', '득점', '안타', '2타', '3타', '홈런',
                     '루타', '타점', '도루', '도실', '볼넷', '사구', '고4', '삼진',
                     '병살', '희타', '희비', '투구', 'barrel', '장타', '출루']

        on_base = ['안타', '2타', '3타', '홈런', '사구', '볼넷', '고4']
        total_base = ['타수', '볼넷', '고4', '사구', '희타']
        hit = ['안타', '2타', '3타', '홈런']

    barell = ['2타', '3타', '홈런']

    on_base = ['안타', '2타', '3타', '홈런', '사구', '볼넷', '고4']
    out = ['병살', '삼진', '희타']
    etc = ['타수', '도루', '투구']

    is_bin = ['선발']

    # for calcaul
    total_base =['타수', '볼넷', '고4', '사구', '희타']
    hit = ['안타', '2타', '3타', '홈런']

    sample_list = []

    for i in range(1, len(df), window_size):
        single = df[-i:-(i + window_size):-1].groupby(['이름', 'year'])[on_base + out + etc].sum()
        single['gap'] = df[-i:-(i + window_size):-1].iloc[0, 3] - df[-i:-(i + window_size):-1].iloc[-1, 3]

        single['num_game'] = df[-i:-(i + window_size):-1].shape[0]
        single['출루'] = single[on_base].sum(axis=1) / single[total_base].sum(axis=1)
        single['장타'] = (single[hit] * [1, 2, 3, 4]).sum(axis=1) / single['타수']
        sample_list.append(single)

    result = pd.concat(sample_list).reset_index()
    result.index = ['t' + str(i) for i in range(1, len(result) + 1)]

    result['출루'] = [None] + list(result['장'].iloc[:-1])
    result['장타'] = [None] + list(result['on_base'].iloc[:-1])

    if slice:
        result['출루'] = [None] + list(result['장'].iloc[:-1])
        result['장타'] = [None] + list(result['on_base'].iloc[:-1])

    return result


def stack_ts(df, window_size, game_base, year, y, t_stamp):
    game = df[df.year == year]
    tot_players = game[(game.gap < game_base)].reset_index().groupby('이름')['index'].count().gt(t_stamp - 1).items()
    players = [name for name, flag in tot_players if flag == True]

    sequence = []
    for name in players:
        sequence.append(game[(game.gap < game_base) & (game['이름'] == name)][y].to_frame().T.iloc[:, :t_stamp])
        sequence[-1].columns = ['t' + str(i) for i in range(0, t_stamp)]

    tot_seq = reduce(lambda left, right: pd.concat([left, right]), sequence)
    tot_seq.index = players
    tot_seq = tot_seq.dropna()
    print(f'{year} year feature {y} ==={window_size} ws ')
    return tot_seq


def stack_ts_interpol(df, window_size, game_base, year, y, t_stamp):
    game = df[df.year == year]
    tot_players = game[(game.gap < game_base)].reset_index().groupby('이름')['index'].count().gt(t_stamp - 4).items()
    players = [name for name, flag in tot_players if flag == True]

    sequence = []
    for name in players:
        sequence.append(game[(game.gap < game_base) & (game['이름'] == name)][y].to_frame().T.iloc[:, :t_stamp])
        sequence[-1].columns = ['t' + str(i) for i in range(0, t_stamp)]

    tot_seq = reduce(lambda left, right: pd.concat([left, right]), sequence)
    tot_seq.index = players
    #     tot_seq=tot_seq.dropna()
    print(f'{year} year feature {y} ==={window_size} ws ')
    return tot_seq