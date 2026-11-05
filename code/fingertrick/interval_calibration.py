import math

def interval_closure(rows, columns, measurements):
    n = rows + columns
    d = [[math.inf] * n for _ in range(n)]
    for v in range(n):
        d[v][v] = 0.0
    for i, j, lower, upper in measurements:
        if not (0 <= i < rows and 0 <= j < columns and (lower <= upper)):
            raise ValueError('Invalid cell interval')
        c = rows + j
        d[c][i] = min(d[c][i], float(upper))
        d[i][c] = min(d[i][c], -float(lower))
    for k in range(n):
        for i in range(n):
            if math.isinf(d[i][k]):
                continue
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    if any((d[v][v] < -1e-10 for v in range(n))):
        raise ValueError('No additive means satisfy these intervals')
    return d

def cell_bounds(distance, rows, row, column):
    c = rows + column
    return (-distance[row][c], distance[c][row])

def choice_regrets(distance, rows, columns):
    return [max((distance[rows + h][rows + k] for k in range(columns))) for h in range(columns)]

def two_column_radius(means):
    if not means or any((len(row) != 2 for row in means)):
        raise ValueError('Expected a nonempty two-column array')
    differences = [row[0] - row[1] for row in means]
    delta = (max(differences) + min(differences)) / 2
    fitted = [[sum(row) / 2 + delta / 2, sum(row) / 2 - delta / 2] for row in means]
    return ((max(differences) - min(differences)) / 4, fitted)
