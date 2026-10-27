import Information from "./information"
import './modal.css'

export default function Modal({closeModal, data, header}) {
    const listItems = Array.isArray(data) ? data : []

    return (
        <div className='modalBackground'>
            <div className='modalContainer'>
                <div className='titleCloseBtn'>
                    <button className='titleCloseBtn' onClick={() => {closeModal(false)}}>❌</button>
                </div>
                <div className='title'>{header}</div>
                <div className='body'>

                    <u1>
                    {listItems.map((item, index) => (
                        <Information id='list-div'
                        key={item.Title || index}
                        rank={item.rank}
                        title={item.Title}
                        price={item.price}
                        rating={item.rating}
                        // genres={}
                        />
                    ))}
                    </u1>
                </div>
            </div>
        </div>
    )
}