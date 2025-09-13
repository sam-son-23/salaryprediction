# save as build_full_salary_dataset.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random
import time

# --------------------
# CONFIG
# --------------------

company_tiers = {
    # Tier 1
    'google': 1, 'microsoft': 1, 'amazon': 1, 'apple': 1, 'meta': 1,
    'netflix': 1, 'linkedin': 1, 'adobe': 1, 'salesforce': 1, 'uber': 1,

    # Tier 2
    'ibm': 2, 'accenture': 2, 'oracle': 2, 'sap': 2, 'deloitte': 2,
    'dell-technologies': 2, 'intel': 2, 'intuit': 2, 'vmware': 2, 'cisco': 2,

    # Tier 3
    'tcs': 3, 'infosys': 3, 'wipro': 3, 'cts': 3, 'hcl-technologies': 3,
    'capgemini': 3, 'persistent': 3, 'tech-mahindra': 3,

    # Tier 4
    'virtusa': 4, 'hexaware': 4, 'zensar': 4, 'birlasoft': 4, 'niit-technologies': 4,
    'sonata-software': 4,

    # Tier 5
    'quest-global': 5, 'valuelabs': 5, 'coforge': 5, 'kpit': 5, 'infogain': 5,
    'happiest-minds': 5,
}

locations = ['Bangalore', 'Hyderabad', 'Chennai', 'Mumbai', 'Pune', 'Delhi', 'Gurgaon', 'Noida', 'Kolkata']
education_levels = ['Bachelor', 'Master', 'PhD']

certifications = [
    'None',
    'AWS Certified Solutions Architect', 'AWS Cloud Practitioner', 'Microsoft Certified: Azure Fundamentals',
    'Google Professional Cloud Architect', 'Google Associate Cloud Engineer',
    'Certified Information Systems Security Professional (CISSP)', 'Certified Scrum Master (CSM)',
    'Project Management Professional (PMP)', 'Certified Ethical Hacker (CEH)',
    'CompTIA Security+', 'ITIL Foundation', 'Cisco Certified Network Associate (CCNA)',
    'Oracle MySQL Database Administration', 'IBM Data Science Professional Certificate',
    'Salesforce Certified Administrator', 'Microsoft Data Analyst Associate',
]

skills = [
    'Cloud', 'AI', 'Security', 'Full Stack', 'Data Engineering', 'DevOps',
    'Data Science', 'Big Data', 'Machine Learning', 'Deep Learning', 'NLP',
    'Cybersecurity', 'AWS', 'Azure', 'Google Cloud', 'MLOps',
    'Front-end Development', 'Back-end Development', 'Mobile Development',
    'Blockchain', 'IoT', 'Database Management', 'QA Automation', 'Network Security',
    'UX/UI Design', 'RPA', 'Technical Writing', 'API Development',
    'Prompt Engineering', 'Edge Computing', 'AR/VR', 'Project Management', 'Agile/Scrum'
]

roles = [
    'Software Developer', 'Senior Developer', 'Tech Lead', 'QA Engineer',
    'Data Scientist', 'DevOps Engineer', 'Cloud Engineer', 'Business Analyst',
    'Machine Learning Engineer', 'Product Manager', 'Security Engineer',
    'AI Researcher', 'Solution Architect', 'Full Stack Developer',
    'Database Administrator', 'UX/UI Designer', 'Project Manager', 'Data Analyst',
    'Business Intelligence Analyst', 'Network Engineer', 'System Administrator',
    'Mobile App Developer', 'Blockchain Developer', 'Site Reliability Engineer',
    'Cybersecurity Specialist', 'Cloud Architect', 'Scrum Master', 'Technical Lead',
    'Performance Engineer', 'Infrastructure Engineer'
]


# --------------------
# SCRAPER FUNCTION
# --------------------

def scrape_ambitionbox(company_slug, max_pages=3):
    """
    Scrapes salary data from AmbitionBox for a given company.
    """
    url_base = f"https://www.ambitionbox.com/salaries/{company_slug}-salaries"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    results = []

    for page in range(1, max_pages + 1):
        url = f"{url_base}?page={page}"
        print(f"Scraping: {url}")
        res = requests.get(url, headers=headers)

        if res.status_code != 200:
            print(f"Failed: {res.status_code}")
            break

        soup = BeautifulSoup(res.text, 'html.parser')
        cards = soup.find_all('div', class_='salary-card')

        if not cards:
            print(f"No data found on page {page}.")
            break

        for card in cards:
            role_tag = card.find('div', class_='name')
            salary_tag = card.find('div', class_='avgSalary')
            exp_tag = card.find('div', class_='expRange')

            if role_tag and salary_tag:
                role = role_tag.text.strip()
                salary_text = salary_tag.text.strip().replace('₹', '').replace(',', '').split(' ')[0]
                exp_text = exp_tag.text.strip() if exp_tag else None

                try:
                    avg_salary = int(salary_text)
                except:
                    continue

                results.append({
                    'company_slug': company_slug,
                    'role': role,
                    'average_salary_k': avg_salary,  # in '000s
                    'experience_range': exp_text
                })

        time.sleep(random.uniform(2, 4))  # polite delay

    return pd.DataFrame(results)

# --------------------
# DATA ENRICHMENT
# --------------------

def enrich_data(df, company_slug):
    enriched_rows = []

    for _, row in df.iterrows():
        current_company = row['company_slug']
        current_role = row['role']
        current_company_tier = company_tiers.get(current_company, 5)

        # Generate other fields
        gender = random.choices(['Male', 'Female', 'Other'], weights=[0.65, 0.33, 0.02])[0]
        location = random.choice(locations)
        education = random.choice(education_levels)
        cert = random.choice(certifications)
        skill = random.choice(skills)

        # Parse experience range
        if isinstance(row['experience_range'], str) and 'yrs' in row['experience_range']:
            exp_parts = [s.strip() for s in row['experience_range'].split('-')]
            try:
                low = float(exp_parts[0])
                high = float(exp_parts[1].split()[0])
                years_exp = round(random.uniform(low, high), 1)
            except:
                years_exp = round(random.uniform(1, 15), 1)
        else:
            years_exp = round(random.uniform(1, 15), 1)

        # Current CTC: scraped avg salary × 1000 + noise (±10%)
        avg_salary = row['average_salary_k'] * 1000
        current_ctc = int(avg_salary * random.uniform(0.9, 1.1))

        # Target company: pick another company from the list
        target_company = random.choice([c for c in company_tiers.keys() if c != current_company])
        target_company_tier = company_tiers.get(target_company, 5)

        # Expected salary: current_ctc + 20% to 50% hike (simulate)
        hike_factor = random.uniform(1.2, 1.5)
        expected_salary = int(current_ctc * hike_factor)

        enriched_rows.append({
            'current_company': current_company,
            'target_company': target_company,
            'years_of_experience': years_exp,
            'current_salary': current_ctc,
            'expected_salary': expected_salary,
            'gender': gender,
            'location': location,
            'current_role': current_role,
            'sector': 'IT',
            'education': education,
            'certifications': cert,
            'skills': skill,
            'current_company_tier': current_company_tier,
            'target_company_tier': target_company_tier
        })

    return pd.DataFrame(enriched_rows)

# --------------------
# MAIN SCRIPT
# --------------------

def build_dataset():
    all_enriched = []

    for company_slug in company_tiers.keys():
        print(f"\n--- Processing {company_slug} ---")
        scraped_df = scrape_ambitionbox(company_slug, max_pages=3)
        print(f"Scraped {len(scraped_df)} rows from {company_slug}")

        enriched_df = enrich_data(scraped_df, company_slug)
        all_enriched.append(enriched_df)

    final_df = pd.concat(all_enriched, ignore_index=True)
    final_df.to_csv('realistic_salary_dataset.csv', index=False)
    print(f"\n✅ Dataset complete! Saved {len(final_df)} rows to 'realistic_salary_dataset.csv'.")


if __name__ == "__main__":
    build_dataset()
