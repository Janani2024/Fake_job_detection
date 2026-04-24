"""
Generates a synthetic fake_job_postings.csv for demo/testing.
Run: python generate_sample_data.py
"""
import pandas as pd
import random
import os

random.seed(42)

REAL_TITLES = [
    "Software Engineer", "Data Analyst", "Product Manager", "Marketing Manager",
    "UX Designer", "DevOps Engineer", "Business Analyst", "HR Manager",
    "Financial Analyst", "Customer Support Specialist", "Sales Executive",
    "Full Stack Developer", "Machine Learning Engineer", "Project Manager",
    "Content Writer", "Graphic Designer", "Network Administrator", "QA Engineer",
]

FAKE_TITLES = [
    "Work From Home - Earn $5000/week", "Online Data Entry - No Experience",
    "Make Money Fast - Flexible Hours", "Home Based Job - Unlimited Earnings",
    "Part Time Online Job - Easy Money", "Earn Daily from Home",
    "Simple Copy Paste Work - High Pay", "Social Media Manager - $200/day",
]

REAL_DESCRIPTIONS = [
    "We are looking for an experienced professional to join our team. The ideal candidate will have strong analytical skills and the ability to work collaboratively. Responsibilities include developing solutions, collaborating with cross-functional teams, and delivering high-quality results. We offer competitive salary, health benefits, and career growth opportunities.",
    "Join our growing company as a key team member. You will be responsible for managing projects, coordinating with stakeholders, and ensuring timely delivery. The role requires excellent communication skills and relevant industry experience. We provide a supportive work environment with opportunities for professional development.",
    "We are seeking a talented individual to contribute to our mission. The position involves working closely with senior leadership, analyzing data, and generating insights. A bachelor's degree and 2+ years of relevant experience are required. We offer full benefits including medical, dental, and 401k.",
    "An exciting opportunity to work in a fast-paced environment. You will manage client relationships, drive business development, and contribute to company growth. Strong interpersonal skills and a results-driven mindset are essential. Competitive compensation and performance bonuses available.",
    "Our company is hiring a motivated professional to lead key initiatives. You'll collaborate with engineering, design, and business teams to deliver impactful outcomes. Proficiency in relevant tools and technologies is required. Benefits include flexible hours, remote work options, and professional training.",
]

REAL_REQUIREMENTS = [
    "Bachelor's degree in Computer Science or related field. 3+ years of experience. Strong problem-solving skills. Proficiency in Python, Java, or similar languages. Excellent communication skills.",
    "Minimum 2 years of relevant experience. Strong analytical and organizational skills. Proficiency in MS Office or Google Workspace. Ability to work independently and in a team.",
    "Bachelor's degree required. Experience with project management tools. Strong written and verbal communication. Attention to detail and ability to meet deadlines.",
    "MBA or equivalent degree preferred. 5+ years of industry experience. Leadership skills and ability to manage teams. Strategic thinking and business acumen.",
    "Technical background with hands-on experience. Familiarity with agile methodologies. Excellent presentation and collaboration skills.",
]

FAKE_DESCRIPTIONS = [
    "Urgent hiring! Work from home opportunity. No experience required. Earn up to $5000 per week easily. Limited positions available. Click the link below and pay a small registration fee to get started immediately. Hurry up before positions are filled!",
    "Make money online from the comfort of your home. No skills needed. We guarantee daily payments. Just register now by paying a small security deposit. Earn $200-$500 per day doing simple online tasks. 100% legitimate! Act now!",
    "Earn big money without any experience or qualifications. This exclusive work from home opportunity offers unlimited income. Send us your bank details and registration fee to start immediately. No interview required. Positions filling up fast!",
    "Online job opportunity: copy paste work, data entry, form filling. Earn Rs 50000/month. No target, no boss. Work from anywhere. Just pay registration fee of Rs 500 to join. Immediate joining. Limited seats!",
    "We are offering a lucrative home-based job. Earn money by liking Facebook posts and sharing links. Daily payment guaranteed. No experience needed. Register now and pay activation fee to get your login credentials.",
]

FAKE_REQUIREMENTS = [
    "No experience required. Only need a smartphone or computer. Must pay registration fee. Must be willing to share personal bank details. No background check needed.",
    "Anyone can apply. Age 18+. No qualifications needed. Just need internet connection. Pay small joining fee to activate account.",
    "Willing to work from home. No degree required. Need to invest a small refundable deposit to start. Immediate starters preferred.",
]

REAL_BENEFITS = [
    "Health insurance, dental, vision. 401k with company match. Flexible working hours. Remote work options. Annual performance bonus. Professional development budget.",
    "Competitive salary and benefits package. Paid time off, sick leave, and holidays. Employee wellness programs. Career advancement opportunities.",
    "Medical and dental coverage. Stock options. Learning and development allowance. Team retreats and company events.",
]

FAKE_BENEFITS = [
    "Unlimited income potential. Work whenever you want. No boss. Be your own boss. Daily payments. Extra bonuses for referrals.",
    "Earn while you sleep. Passive income. No investment limit. Referral bonuses up to $1000 per person.",
    "",
]


def make_real():
    return {
        "title": random.choice(REAL_TITLES),
        "company_profile": "Established company with a strong track record in the industry.",
        "description": random.choice(REAL_DESCRIPTIONS),
        "requirements": random.choice(REAL_REQUIREMENTS),
        "benefits": random.choice(REAL_BENEFITS),
        "fraudulent": 0,
    }


def make_fake():
    return {
        "title": random.choice(FAKE_TITLES),
        "company_profile": "",
        "description": random.choice(FAKE_DESCRIPTIONS),
        "requirements": random.choice(FAKE_REQUIREMENTS),
        "benefits": random.choice(FAKE_BENEFITS),
        "fraudulent": 1,
    }


if __name__ == "__main__":
    records = []
    # ~85% real, 15% fake — mirrors real-world distribution
    for _ in range(850):
        records.append(make_real())
    for _ in range(150):
        records.append(make_fake())

    random.shuffle(records)
    df = pd.DataFrame(records)
    out = os.path.join("data", "fake_job_postings.csv")
    os.makedirs("data", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")
    print(f"Real: {(df.fraudulent==0).sum()}  Fake: {(df.fraudulent==1).sum()}")
