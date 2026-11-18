"""Build the Ukrainian report from measured JSON files, never invented metrics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT=Path(__file__).resolve().parent


def build(repo_url):
    read=lambda name:json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
    meta=read('training_metadata.json')
    progress=read('training_progress.json')
    final=read('evaluation.json')
    random=read('random.json')
    initial=read('initial.json')
    selected=meta['selected_round']
    selected_history=[h for h in progress['history'] if h['round']==selected]
    fig,axes=plt.subplots(1,2,figsize=(10.4,2.65),layout='constrained')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    for v in progress['validation']:
        hist=[h for h in progress['history'] if h['round']==v['round']]
        axes[0].plot(range(1,16),[h['loss'] for h in hist],label=f"Раунд {v['round']}")
    axes[0].set(xlabel='Епоха',ylabel='Cross-entropy',title='Фактична помилка навчання')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    rows=final['episodes']
    axes[1].bar(range(1,len(rows)+1),[r['return_'] for r in rows],color='#147d87')
    axes[1].axhline(0,color='#24364a',linewidth=.7)
    axes[1].set(xlabel='Номер тестової партії',ylabel='Наші очки мінус очки суперника',title='Фінальний тест: 20 нових seeds',ylim=(-22,22))
    axes[1].set_xticks([1,5,10,15,20])
    axes[1].grid(axis='y',alpha=.2)
    chart=ROOT/'results/learning_results.png'
    fig.savefig(chart,dpi=180)
    plt.close(fig)
    pdfmetrics.registerFont(TTFont('DV',findfont('DejaVu Sans')))
    pdfmetrics.registerFont(TTFont('DV-Bold',findfont('DejaVu Sans:weight=bold')))
    pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyUA',fontName='DV',fontSize=9.3,leading=13,spaceAfter=7,textColor=colors.HexColor('#223044')))
    styles.add(ParagraphStyle(name='TitleUA',fontName='DV-Bold',fontSize=21,leading=26,spaceAfter=10,textColor=colors.HexColor('#102e45')))
    styles.add(ParagraphStyle(name='HUA',fontName='DV-Bold',fontSize=12,leading=16,spaceBefore=10,spaceAfter=7,textColor=colors.HexColor('#147d87')))
    styles.add(ParagraphStyle(name='SmallUA',fontName='DV',fontSize=8,leading=11,spaceAfter=5,textColor=colors.HexColor('#34485a')))
    body=[]
    def p(text,style='BodyUA'):return Paragraph(text,styles[style])
    def add(text,style='BodyUA'):body.append(p(text,style))
    def table(data,widths):
        t=Table([[p(str(c),'SmallUA') for c in row] for row in data],colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f1f3')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),
                              ('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#147d87')),
                              ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d6e0e5'))]))
        body.append(t)
        body.append(Spacer(1,2*mm))
    add('ЛАБОРАТОРНА РОБОТА · 18.11.2026','SmallUA')
    add('Навчання нейронної мережі<br/>грати в Atari Pong','TitleUA')
    add('<b>Дисципліна:</b> Програмування нейронних мереж<br/><b>Виконав:</b> Личак Дмитро Романович, група БЗ-123-23-1-СП<br/><b>Викладач:</b> Юрій Віталійович Дегтяр')
    add('1. Мета та середовище','HUA')
    add('Створити й навчити нейронну мережу перемагати у грі Atari. Обрано Pong у справжньому емуляторі Arcade Learning Environment: агент керує правою ракеткою, суперником керує гра. Перемога: 21 вигране очко та підтверджене середовищем завершення гри. Нагорода +1 за вигране очко, -1 за пропущене.')
    add('Конфігурація: ALE/Pong-v5, frameskip=1, імовірність повторення попередньої дії 0,25; ліміт 108 000 кроків. Це власний протокол оцінювання, а не стандартний Atari benchmark із frameskip=4.')
    add('2. Архітектура та метод','HUA')
    add('Використано навчання за демонстраціями: behavioral cloning та два раунди DAgger. RGB-обробник знаходить м’яч і ракетку за кольором, оцінює швидкість за двома кадрами та аналітично прогнозує точку перехоплення з урахуванням відскоків. Мережа отримує 7 ознак: x, y, vx, vy м’яча, y ракетки, видимість м’яча та прогнозовану y-координату.')
    table([['Компонент','Реалізація'],['Мережа',f"MLP 7 → 64 → 64 → 3; ReLU; {meta['parameters']:,} параметрів"],['Вихід','Три оцінки дій: стояти, вгору, вниз; на тесті argmax'],['Навчання','Cross-entropy; Adam; lr=0,001; batch=1024; 15 епох/раунд'],['Дані',f"8 навчальних партій, {meta['samples']:,} пар «ознаки–дія»"],['Обчислення',f"CPU; Python {meta['python']}; PyTorch {meta['torch']}; seed=42"]],[43*mm,131*mm])
    add('Демонстратор порівнює прогнозовану точку перехоплення з положенням ракетки (зона спокою ±2 пікселі). Спочатку зібрано 4 його партії; далі по 2 партії для кожного раунду DAgger. У першому раунді DAgger частка дій демонстратора становить 20%, у другому всі дії виконує мережа. Нові розмічені стани додаються до спільного набору.')
    add('<b>Межа методу:</b> мережа навчається керуванню з підготовлених ознак. Детектор і фізичний прогноз задані вручну. Це не PPO, не DQN і не наскрізне навчання гри з пікселів. Ігрова нагорода служить метрикою, а навчальна ціль - правильна дія демонстратора. Під час тестування демонстратор не викликається.','SmallUA')
    body.append(PageBreak())
    add('3. Перевірка та результати','HUA')
    add(f"Навчальні seeds: 100–103, 200–201, 300–301. Після кожного раунду виконано перевірку на seeds 1000–1002. За кількістю перемог, а потім середньою нагородою обрано <b>раунд {selected}</b>. Обрана модель навчалась на {selected_history[-1]['samples']:,} прикладах. Вибір зроблено до фінального тесту на 20 інших seeds 10000–10019. На тесті ваги не змінювалися.")
    def result_row(name,result):
        s=result['summary']
        return [name,f"{s['wins']} / {s['episodes']}",f"{s['mean_return']:.2f}",f"{s['std_return']:.2f}",str(s['truncated'])]
    table([['Політика','Перемоги','Сер. нагорода','Станд. відх.','Ліміт часу'],
           result_row('Випадкові дії',random),result_row('Початкова мережа',initial),result_row('Навчена мережа',final)],
          [49*mm,27*mm,35*mm,33*mm,30*mm])
    body.append(Spacer(1,5*mm))
    body.append(Image(str(chart),width=174*mm,height=44.34*mm))
    s=final['summary'];lo,hi=s['win_rate_wilson95']
    add(f"<b>Результат:</b> {s['wins']} перемог із {s['episodes']} ({s['win_rate']*100:.0f}%). Середня різниця рахунку {s['mean_return']:+.2f}; діапазон від {s['min_return']:+d} до {s['max_return']:+d}. Приблизний 95% інтервал Вілсона для частки перемог: {lo*100:.1f}–{hi*100:.1f}%. Випадки завершення за лімітом часу: {s['truncated']}; вони не рахуються перемогами.")
    val='; '.join(f"раунд {v['round']}: {v['summary']['wins']}/3, нагорода {v['summary']['mean_return']:.2f}" for v in progress['validation'])
    add('Проміжна перевірка: '+val+'. Три партії на етапі вибору моделі є малою вибіркою. Фінальні 20 партій дають окрему оцінку, але не гарантію виграшу в усіх можливих іграх.','SmallUA')
    if selected==0:
        add('Для фінального тесту обрано початкове behavioral cloning. Подальші раунди DAgger зменшили помилку класифікації, але не поліпшили кількість завершених перемог на перевірці.','SmallUA')
    add('4. Висновок і відтворення','HUA')
    add('Навчання змінило поведінку мережі порівняно з випадковими початковими вагами. Досягнутий результат стосується Pong із зазначеними налаштуваннями та ручним прогнозом траєкторії. Код, ваги, навчальні приклади й результати кожної партії збережено для перевірки. Проєкт SnakeAI використано як методичний приклад; його код і ваги не копіювалися.')
    add('<b>Команди після встановлення requirements.txt:</b><br/>python pong_lab.py evaluate<br/>python pong_lab.py play --video results/demo.mp4<br/>python pong_lab.py train &nbsp; (повторне навчання з нуля)','SmallUA')
    add(f'<b>Код на GitHub:</b> <link href="{repo_url}" color="#147d87">{repo_url}</link>','SmallUA')
    add('Джерела','HUA')
    add('1. Ю. В. Дегтяр. TrainAI_Snake: <link href="https://github.com/YuriiDehtiar/TrainAI_Snake">github.com/YuriiDehtiar/TrainAI_Snake</link>; надана презентація SnakeAI v9.<br/>2. Farama Foundation. Pong: <link href="https://ale.farama.org/environments/pong/">ale.farama.org/environments/pong/</link>.<br/>3. Ross, Gordon, Bagnell (2011). DAgger: <link href="https://arxiv.org/abs/1011.0686">arxiv.org/abs/1011.0686</link>.','SmallUA')
    def footer(canvas,doc):
        canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#667788'))
        canvas.drawString(18*mm,11*mm,'Личак Дмитро · БЗ-123-23-1-СП · Програмування нейронних мереж')
        canvas.drawRightString(192*mm,11*mm,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=(210*mm,297*mm),
                          leftMargin=18*mm,rightMargin=18*mm,topMargin=15*mm,bottomMargin=18*mm,
                          title='Atari Pong — лабораторна робота',author='Личак Дмитро Романович')
    doc.build(body,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo-url',required=True)
    build(parser.parse_args().repo_url)
