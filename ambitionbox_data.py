# save as ambitionbox_data.py - Enhanced with 70+ skills and 50+ certifications

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

# 50+ Certifications for 2025
certifications = [
    'None',
    'AWS Certified Solutions Architect - Associate',
    'AWS Certified Solutions Architect - Professional', 
    'AWS Certified Developer - Associate',
    'AWS Certified SysOps Administrator - Associate',
    'AWS Certified DevOps Engineer - Professional',
    'AWS Certified Security - Specialty',
    'AWS Certified Machine Learning - Specialty',
    'AWS Certified Data Analytics - Specialty',
    'AWS Cloud Practitioner',
    'Microsoft Certified: Azure Fundamentals',
    'Microsoft Certified: Azure Administrator Associate',
    'Microsoft Certified: Azure Solutions Architect Expert',
    'Microsoft Certified: Azure DevOps Engineer Expert',
    'Microsoft Certified: Azure Security Engineer Associate',
    'Microsoft Certified: Azure AI Engineer Associate',
    'Microsoft Certified: Azure Data Scientist Associate',
    'Microsoft Certified: Azure Data Engineer Associate',
    'Google Cloud Professional Cloud Architect',
    'Google Cloud Professional Data Engineer',
    'Google Cloud Professional Machine Learning Engineer',
    'Google Cloud Professional Security Engineer',
    'Google Cloud Associate Cloud Engineer',
    'Google Cloud Professional DevOps Engineer',
    'Certified Information Systems Security Professional (CISSP)',
    'Certified Information Security Manager (CISM)',
    'Certified Information Systems Auditor (CISA)',
    'Certified Ethical Hacker (CEH)',
    'CompTIA Security+',
    'CompTIA Network+',
    'CompTIA A+',
    'CompTIA Cloud+',
    'CompTIA CySA+',
    'Cisco Certified Network Associate (CCNA)',
    'Cisco Certified Network Professional (CCNP)',
    'Cisco Certified Internetwork Expert (CCIE)',
    'Project Management Professional (PMP)',
    'Certified Scrum Master (CSM)',
    'Professional Scrum Master (PSM)',
    'SAFe Agilist Certification',
    'ITIL Foundation',
    'ITIL Expert',
    'Docker Certified Associate (DCA)',
    'Certified Kubernetes Administrator (CKA)',
    'Certified Kubernetes Application Developer (CKAD)',
    'Certified Kubernetes Security Specialist (CKS)',
    'TensorFlow Developer Certificate',
    'Oracle Database Administrator Certified Professional',
    'Oracle MySQL Database Administration',
    'Salesforce Certified Administrator',
    'Salesforce Certified Platform Developer',
    'Certified Cloud Security Professional (CCSP)',
    'Certificate of Cloud Security Knowledge (CCSK)',
    'VMware Certified Professional (VCP)',
    'Red Hat Certified Engineer (RHCE)',
    'Certified Data Professional (CDP)'
]

# 70+ Skills for 2025
skills = [
    'Cloud Computing', 'Artificial Intelligence', 'Machine Learning', 'Deep Learning', 'Data Science',
    'Cybersecurity', 'DevOps', 'Full Stack Development', 'Data Engineering', 'Big Data',
    'Natural Language Processing', 'Computer Vision', 'MLOps', 'Cloud Security', 'AWS',
    'Microsoft Azure', 'Google Cloud Platform', 'Kubernetes', 'Docker', 'Terraform',
    'Ansible', 'Jenkins', 'Python', 'Java', 'JavaScript', 'TypeScript', 'React',
    'Angular', 'Vue.js', 'Node.js', 'Spring Boot', 'Django', 'Flask', 'FastAPI',
    'Microservices', 'REST APIs', 'GraphQL', 'Database Management', 'SQL', 'NoSQL',
    'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'ElasticSearch', 'Apache Kafka',
    'Apache Spark', 'Hadoop', 'Tableau', 'Power BI', 'Data Visualization', 'Blockchain',
    'Cryptocurrency', 'Smart Contracts', 'IoT', 'Edge Computing', 'Serverless Architecture',
    'Lambda Functions', 'API Gateway', 'Message Queues', 'Event-Driven Architecture',
    'CI/CD', 'Git', 'GitHub Actions', 'GitLab CI', 'Agile', 'Scrum', 'Kanban',
    'Project Management', 'Technical Writing', 'Code Review', 'Test Automation',
    'Unit Testing', 'Integration Testing', 'Performance Testing', 'Load Testing',
    'Security Testing', 'Penetration Testing', 'Vulnerability Assessment', 'Network Security',
    'Information Security', 'Identity Management', 'OAuth', 'SAML', 'Single Sign-On'
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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    results = []
    for page in range(1, max_pages + 1):
        url = f"{url_base}?page={page}"
        print(f"Scraping: {url}")
        
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                print(f"Failed: {res.status_code}")
                break
            
            soup = BeautifulSoup(res.text, 'html.parser')
            cards = soup.find_all('div', class_='salary-card')
            
            if not cards:
                print(f"No data found on page {page}.")
                break
            
            for card in cards:
                try:
                    role_tag = card.find('div', class_='name')
                    salary_tag = card.find('div', class_='avgSalary')
                    exp_tag = card.find('div', class_='expRange')
                    
                    if role_tag and salary_tag:
                        role = role_tag.text.strip()
                        salary_text = salary_tag.text.strip().replace('₹', '').replace(',', '').split(' ')[0]
                        exp_text = exp_tag.text.strip() if exp_tag else None
                        
                        try:
                            avg_salary = int(salary_text)
                        except (ValueError, TypeError):
                            continue
                        
                        results.append({
                            'company_slug': company_slug,
                            'role': role,
                            'average_salary_k': avg_salary,  # in '000s
                            'experience_range': exp_text
                        })
                        
                except Exception as e:
                    print(f"Error parsing card: {e}")
                    continue
            
            time.sleep(random.uniform(2, 4))  # polite delay
            
        except Exception as e:
            print(f"Error scraping page {page}: {e}")
            break
    
    return pd.DataFrame(results)

# --------------------
# DATA ENRICHMENT
# --------------------

def enrich_data(df, company_slug):
    """
    Enriches scraped data with additional features using enhanced skills and certifications
    """
    enriched_rows = []
    
    for _, row in df.iterrows():
        current_company = row['company_slug']
        current_role = row['role']
        current_company_tier = company_tiers.get(current_company, 5)
        
        # Generate other fields with realistic distributions
        gender = random.choices(['Male', 'Female', 'Other'], weights=[0.65, 0.33, 0.02])[0]
        location = random.choice(locations)
        education = random.choice(education_levels)
        
        # Select certification based on role (more realistic assignment)
        if 'Cloud' in current_role or 'DevOps' in current_role:
            cert = random.choice([cert for cert in certifications if 'AWS' in cert or 'Azure' in cert or 'Google Cloud' in cert])
        elif 'Security' in current_role or 'Cyber' in current_role:
            cert = random.choice([cert for cert in certifications if 'Security' in cert or 'CISSP' in cert or 'CEH' in cert])
        elif 'Data' in current_role or 'ML' in current_role or 'AI' in current_role:
            cert = random.choice([cert for cert in certifications if 'Data' in cert or 'ML' in cert or 'TensorFlow' in cert])
        else:
            cert = random.choice(certifications)
        
        # Select skill based on role (more realistic assignment)
        if 'Cloud' in current_role:
            skill = random.choice(['Cloud Computing', 'AWS', 'Microsoft Azure', 'Google Cloud Platform', 'Kubernetes', 'Docker'])
        elif 'Data Scientist' in current_role or 'ML' in current_role:
            skill = random.choice(['Machine Learning', 'Data Science', 'Python', 'Deep Learning', 'Natural Language Processing'])
        elif 'DevOps' in current_role:
            skill = random.choice(['DevOps', 'CI/CD', 'Jenkins', 'Docker', 'Kubernetes', 'Terraform'])
        elif 'Security' in current_role:
            skill = random.choice(['Cybersecurity', 'Network Security', 'Penetration Testing', 'Information Security'])
        elif 'Full Stack' in current_role or 'Developer' in current_role:
            skill = random.choice(['Full Stack Development', 'JavaScript', 'React', 'Node.js', 'Python', 'Java'])
        else:
            skill = random.choice(skills)
        
        # Parse experience range
        if isinstance(row['experience_range'], str) and 'yrs' in row['experience_range']:
            exp_parts = [s.strip() for s in row['experience_range'].split('-')]
            try:
                low = float(exp_parts[0])
                high = float(exp_parts[1].split()[0])
                years_exp = round(random.uniform(low, high), 1)
            except (ValueError, IndexError):
                years_exp = round(random.uniform(1, 15), 1)
        else:
            years_exp = round(random.uniform(1, 15), 1)
        
        # Current CTC: scraped avg salary × 1000 + noise (±10%)
        avg_salary = row['average_salary_k'] * 1000
        current_ctc = int(avg_salary * random.uniform(0.9, 1.1))
        
        # Target company: pick another company from the list
        available_companies = [c for c in company_tiers.keys() if c != current_company]
        target_company = random.choice(available_companies)
        target_company_tier = company_tiers.get(target_company, 5)
        
        # Expected salary: current_ctc + 20% to 50% hike (simulate job switch)
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
    """
    Main function to build the enhanced dataset
    """
    print("🚀 Building Enhanced Dataset with 70+ Skills & 50+ Certifications")
    print("=" * 70)
    
    all_enriched = []
    
    for company_slug in company_tiers.keys():
        print(f"\n--- Processing {company_slug.upper()} ---")
        scraped_df = scrape_ambitionbox(company_slug, max_pages=3)
        print(f"Scraped {len(scraped_df)} rows from {company_slug}")
        
        if len(scraped_df) > 0:
            enriched_df = enrich_data(scraped_df, company_slug)
            all_enriched.append(enriched_df)
            print(f"Enriched to {len(enriched_df)} rows with enhanced features")
        else:
            print(f"No data scraped for {company_slug}, skipping...")
    
    if all_enriched:
        final_df = pd.concat(all_enriched, ignore_index=True)
        final_df.to_csv('realistic_salary_dataset.csv', index=False)
        
        print("\n" + "=" * 70)
        print("✅ DATASET COMPLETE!")
        print(f"📊 Total records: {len(final_df):,}")
        print(f"🎯 Skills available: {len(skills)}")
        print(f"🎓 Certifications available: {len(certifications)}")
        print(f"📍 Locations covered: {len(locations)}")
        print(f"💾 Saved as: 'realistic_salary_dataset.csv'")
        print("=" * 70)
    else:
        print("\n❌ No data was successfully scraped. Check your internet connection and try again.")

if __name__ == "__main__":
    build_dataset()