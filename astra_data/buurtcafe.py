#!/usr/bin/env python3
"""Zelfstandige synthetische kassagenerator; Python 3.10+, numpy en pandas.
Gebruik: python buurtcafe.py [--seed INTEGER] [--output PAD.csv]
Geen netwerk, externe bestanden of imports uit de andere generatoren.
Alle bedragen zijn EUR inclusief btw; synthetische aannames, geen gemeten data.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ['order_id', 'tijdstip', 'dagdeel', 'aantal_gasten', 'item',
           'categorie', 'aantal', 'stukprijs', 'regelbedrag']
PARTS = ['ochtend', 'lunch', 'middag', 'diner', 'avond']
# Lokale, tijdzone-naieve kalenderdatum; geen boekingen over middernacht.
WINDOWS = [(6, 11), (11, 14), (14, 17), (17, 21), (21, 24)]
PREFIX = 'BC'
DEFAULT_SEED = 917203
MENU = [('Koffie', 'warm', 300), ('Cappuccino', 'warm', 340), ('Thee', 'warm', 300), ('Cola', 'fris', 330), ('Bruiswater', 'fris', 290), ('IJsthee', 'fris', 350), ('Pils', 'alcohol', 340), ('Speciaalbier', 'alcohol', 540), ('Huiswijn', 'alcohol', 510), ('Bitterballen', 'snack', 780), ('Nachos', 'snack', 950), ('Kaasplank', 'snack', 1250), ('Tosti', 'gerecht', 650), ('Burger', 'gerecht', 1650), ('Dagschotel', 'gerecht', 1790)]
BASE_GROUPS = [[1, 1, 0, 1, 1], [1, 1, 0.3, 1, 2], [1, 1.2, 2, 2, 0.7], [0.3, 1, 4, 2, 1.5], [0.1, 0.7, 7, 3, 0.15]]
GUEST_P = [0.18, 0.4, 0.23, 0.12, 0.05, 0.02]
CAPACITY = np.array([0, 55, 100, 95, 155])  # gasten per dagdeel, inclusief tafelomloop


def overdispersed(rng, mean, shape):
    return int(rng.poisson(rng.gamma(shape, max(float(mean), 0.0) / shape)))


def season(d):
    # Jaarfase op maand/dag, zodat schrikkeljaren geen fasebreuk geven.
    return np.cos(2 * np.pi * (d.dayofyear - 205) / (366 if d.is_leap_year else 365))


TIME_CENTERS = [9, 12.5, 15.6, 19.3, 22.2]
TIME_WIDTHS = [0.6, 0.5, 0.8, 0.9, 0.8]
MEAL_CHANCE = [0.1, 0.7, 0.45, 0.65, 0.4]
LINE_P = [0.2, 0.4, 0.27, 0.13]
ROUND_CHANCE = 0.38
PRICE_CHANCE = 0.19
GAP_DAYS = 5
GAP_BLOCK = 2

def demand(dates, rng):
    result = []
    anomaly = 0.0
    habit = 0.0
    wet = False
    for i,d in enumerate(dates):
        anomaly = .79 * anomaly + rng.normal(0, 2.5)
        temp = 12 + 9 * season(d) + anomaly
        wet = rng.random() < (.64 if wet else .28)
        habit = .90 * habit + rng.normal(0,.055)
        patio = (max(0, min(temp-15, 10)) * (0 if wet else 1))
        if temp > 29:
            patio *= .35
        local = [0,.68,.80,.98,1.6,1.85,1.18][d.weekday()]
        closed = d.weekday() == 0 or (d.month,d.day) in [(1,1),(12,25)]
        if d.month == 8 and 5 <= d.day <= 14:
            closed = True
        quiz = d.weekday() == 3 and 8 <= d.day <= 14
        derby = d.weekday() == 6 and rng.random() < .16
        shock = rng.lognormal(-.035,.26) * np.exp(habit)
        means = np.array([0,5,11+patio*1.3,14+patio*.45,25]) * local * shock
        means[4] += 15 * quiz + 22 * derby
        means[2:4] *= .70 if wet and temp < 7 else 1
        orders = [0]*5 if closed else [overdispersed(rng,m,14) for m in means]
        result.append(dict(orders=orders, temp=temp, wet=wet))
    return result


def adjust_basket(w, part, ctx):
    if ctx['temp'] > 21 and not ctx['wet']:
        w[1:3] *= 1.5
        w[0] *= .6
    return w


def generate(seed=DEFAULT_SEED, output=None):
    # Gescheiden stromen: wijzigingen aan een mandje veranderen de vraag niet.
    streams = np.random.SeedSequence(seed).spawn(4)
    rng, rb, rp, rq = [np.random.default_rng(s) for s in streams]
    dates = pd.date_range('2023-09-01', '2026-08-31', freq='D')
    daily = demand(dates, rng)
    months = pd.period_range('2023-09', '2026-08', freq='M')
    # Geleidelijke, per product asynchrone prijsaanpassingen in hele centen.
    prices = {}
    for name, category, cents in MENU:
        current = cents
        path = []
        for m in range(len(months)):
            if m and rp.random() < PRICE_CHANCE:
                current = int(5 * round(current * (1 + rp.uniform(.006, .027)) / 5))
            path.append(current)
        prices[name] = path
    menu_groups = {}
    for product in MENU:
        menu_groups.setdefault(product[1], []).append(product)

    output = Path(output or Path(__file__).with_suffix('.csv'))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)
    sequence = 0
    exported = 0
    # Exportstoringen laten alleen hele orders weg, nooit losse orderregels.
    gaps = set(rq.choice(np.arange(1, len(dates)-1), size=GAP_DAYS, replace=False).tolist())
    block_start = int(rq.integers(100, len(dates)-100))
    gaps.update(range(block_start, block_start + GAP_BLOCK))
    for i, (d, context) in enumerate(zip(dates, daily)):
        if i in gaps:
            continue
        partial = rq.random() < .018 and i not in (0, len(dates)-1)
        retention = rq.uniform(.35, .8) if partial else 1.0
        rows = []
        month = (d.year - 2023) * 12 + d.month - 9
        for part_idx, proposed in enumerate(context['orders']):
            guests_used = 0
            for _ in range(int(proposed)):
                guests = int(rb.choice(np.arange(1, 7), p=GUEST_P))
                if guests_used + guests > CAPACITY[part_idx]:
                    break
                guests_used += guests
                sequence += 1
                lo, hi = WINDOWS[part_idx]
                # Piek rond een dagspecifiek centrum, gemengd met inloop.
                center = TIME_CENTERS[part_idx] + context.get('time_shift', 0)
                hour = (rb.uniform(lo, hi) if rb.random() < .20 else
                        np.clip(rb.normal(center, TIME_WIDTHS[part_idx]), lo, hi - 1/3600))
                timestamp = d + pd.Timedelta(seconds=int(hour * 3600))
                weights = np.array(BASE_GROUPS[part_idx], dtype=float)
                weights = adjust_basket(weights, part_idx, context)
                weights /= weights.sum()
                group_names = list(menu_groups)
                # Eén mandje per bestelling, 2-5 verschillende producten.
                nlines = int(rb.choice([2, 3, 4, 5], p=LINE_P))
                selected = []
                # Eerste regel is steeds een passende drank; vervolgens vaak eten.
                drink_choices = [j for j, g in enumerate(group_names) if g in ('warm', 'fris', 'alcohol')]
                dw = weights[drink_choices]; dw /= dw.sum()
                first_group = group_names[int(rb.choice(drink_choices, p=dw))]
                selected.append(menu_groups[first_group][int(rb.integers(len(menu_groups[first_group])))])
                if rb.random() < MEAL_CHANCE[part_idx]:
                    food = [j for j,g in enumerate(group_names) if g not in ('warm','fris','alcohol') and weights[j] > 0]
                    fw = weights[food]; fw /= fw.sum()
                    g = group_names[int(rb.choice(food, p=fw))]
                    selected.append(menu_groups[g][int(rb.integers(len(menu_groups[g])))])
                while len(selected) < nlines:
                    available = [p for p in MENU if p not in selected and weights[group_names.index(p[1])] > 0]
                    pw = np.array([weights[group_names.index(p[1])] / len(menu_groups[p[1]]) for p in available])
                    pw /= pw.sum()
                    selected.append(available[int(rb.choice(len(available), p=pw))])
                keep = rq.random() < retention
                for name, category, _ in selected:
                    # Hoeveelheden delen de groepsgrootte; barbestellingen kunnen rondes bevatten.
                    q = 1 + int(rb.binomial(max(guests - 1, 0), .70 if category in ('warm','fris','alcohol') else .48))
                    if category == 'alcohol' and part_idx >= 3:
                        q += int(rb.binomial(guests, ROUND_CHANCE))
                    if category in ('snack','gebak','dessert'):
                        q = max(1, int(np.ceil(q / 2)))
                    cents = prices[name][month]
                    if keep:
                        rows.append((f'{PREFIX}-{sequence:09d}', timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                     PARTS[part_idx], guests, name, category, q,
                                     f'{cents / 100:.2f}', f'{q * cents / 100:.2f}'))
        if rows:
            frame = pd.DataFrame(rows, columns=COLUMNS)
            frame = frame.sort_values(['tijdstip', 'order_id'], kind='stable')
            frame.to_csv(output, mode='a', header=False, index=False)
            exported += len(frame)
    print(f'{output}: {exported:,} orderregels; seed={seed}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed moet niet-negatief zijn')
    generate(args.seed, args.output)


if __name__ == '__main__':
    main()
