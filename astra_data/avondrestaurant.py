#!/usr/bin/env python3
"""Zelfstandige synthetische kassagenerator; Python 3.10+, numpy en pandas.
Gebruik: python avondrestaurant.py [--seed INTEGER] [--output PAD.csv]
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
PREFIX = 'AR'
DEFAULT_SEED = 239857
MENU = [('Espresso', 'warm', 320), ('Cappuccino', 'warm', 380), ('Thee', 'warm', 330), ('Tafelwater', 'fris', 495), ('Frisdrank', 'fris', 375), ('Alcoholvrije cocktail', 'fris', 850), ('Glas wijn', 'alcohol', 650), ('Speciaalbier', 'alcohol', 575), ('Cocktail', 'alcohol', 1150), ('Aperitief', 'alcohol', 750), ('Steak', 'gerecht', 2950), ('Visgerecht', 'gerecht', 2750), ('Vegetarisch gerecht', 'gerecht', 2250), ('Chefsmenu', 'gerecht', 4450), ('Olijven', 'snack', 550), ('Brood met dips', 'snack', 650), ('Borrelhapjes', 'snack', 950), ('Dessert', 'dessert', 925), ('Kaasselectie', 'dessert', 1250)]
BASE_GROUPS = [[1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0.3, 2, 5, 7, 1, 1.5], [0.6, 1, 8, 0.4, 3, 1]]
GUEST_P = [0.1, 0.49, 0.19, 0.14, 0.06, 0.02]
CAPACITY = np.array([0, 0, 0, 150, 95])  # gasten per dagdeel, inclusief tafelomloop


def overdispersed(rng, mean, shape):
    return int(rng.poisson(rng.gamma(shape, max(float(mean), 0.0) / shape)))


def season(d):
    # Jaarfase op maand/dag, zodat schrikkeljaren geen fasebreuk geven.
    return np.cos(2 * np.pi * (d.dayofyear - 205) / (366 if d.is_leap_year else 365))


TIME_CENTERS = [9, 12.5, 15, 19, 22.3]
TIME_WIDTHS = [0.5, 0.5, 0.5, 0.62, 0.65]
MEAL_CHANCE = [0.1, 0.1, 0.1, 0.99, 0.42]
LINE_P = [0.06, 0.25, 0.42, 0.27]
ROUND_CHANCE = 0.52
PRICE_CHANCE = 0.18
GAP_DAYS = 5
GAP_BLOCK = 2

def demand(dates, rng):
    result=[]
    reputation=0.
    bookings=.0
    for i,d in enumerate(dates):
        reputation=.98*reputation+rng.normal(0,.02)
        bookings=.68*bookings+rng.normal(0,.11)
        weekday=[0,0,.52,.78,1.3,1.55,.82][d.weekday()]
        occasion=(d.month,d.day) in [(2,14),(12,24),(12,26),(12,31)]
        winter=1.12 if d.month in (11,12) else (.83 if d.month in (7,8) else 1)
        # Begrensde reserveringen met samenhangende no-shows, plus losse barinloop.
        demand_level=np.clip(.53*weekday*winter*np.exp(reputation+bookings)+.40*occasion,0,.99)
        booked=rng.binomial(53,demand_level)
        show_probability=rng.beta(44,3)
        seated=rng.binomial(booked,show_probability)
        walkins=overdispersed(rng,4*weekday,5)
        bar_night=rng.random() < (.36 if d.weekday() in (4,5) else .07)
        bar=overdispersed(rng,(8+19*bar_night)*weekday,6)
        orders=[0,0,0,seated+walkins,bar]
        closed=d.weekday() in (0,1) and not occasion
        closed |= d.month == 8 and 3<=d.day<=16
        if closed:
            orders=[0]*5
        result.append(dict(orders=orders,bar_night=bar_night,occasion=occasion,
                           time_shift=.12 if d.weekday() in (4,5) else -.1))
    return result


def adjust_basket(w, part, ctx):
    if ctx['bar_night'] or ctx['occasion']:
        w[2] *= 1.8
    if ctx['occasion']:
        w[5] *= 1.5
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
